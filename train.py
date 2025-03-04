import argparse
import os, time
import json
import shutil
from pydicom import Dataset
from scipy import io
import torch
import numpy as np
import torch.distributed as dist
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from model.model import PETUNet
from model.loss import MSESSIMLoss

EPOCHES = 300
IMAGE_SIZE = 256

class TrainDataset(Dataset):
    def __init__(self, input, target, transform = None):
        self.transform = transform
        input_2 = np.array([input +"/"+ x  for x in os.listdir(input)])
        target_forward = np.array([target +"/"+ x  for x in os.listdir(target)])
        
        assert len(input_2) == len(target_forward)
        
        input_2.sort()
        target_forward.sort()

        self.data = {'input': input_2, 'target': target_forward}
            
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, idx):
        input_path = self.data['input'][idx]
        target_path = self.data['target'][idx]
        
        input_img = io.loadmat(input_path)['img'].astype('float32')
        target_img = io.loadmat(target_path)['img'].astype('float32')
        
        input_img = np.expand_dims(input_img, axis=0)  # (1, H, W)
        target_img = np.expand_dims(target_img, axis=0)  # (1, H, W)
        
        input_img = torch.from_numpy(input_img).float()
        target_img = torch.from_numpy(target_img).float()

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
        
        sample = {
            'input_img': input_img,
            'target_img': target_img,
        }
        return sample

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../models/model_default", help="Path to save checkpoint.")
    parser.add_argument("--input", type=str, default="../mat/NAC_train", help="Input images.")
    parser.add_argument("--target", type=str, default="../mat/CTAC_train", help="Target images.")
    parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
    parser.add_argument("--loss", type=str, default="L2", choices=["L1", "L2"], help="Choose which loss function to use. ")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args

def init_status(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output+"/checkpoint", exist_ok=True)
    with open(args.output+"/commandline_args.yaml" , 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def setup():
    """初始化分布式训练环境"""
    dist.init_process_group("nccl", init_method="env://")  # NCCL 后端（最快）

def cleanup():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def main(world_size, args):
    # 初始化 DDP
    setup()
    rank = dist.get_rank()
    torch.cuda.set_device(rank % world_size)
    device = torch.device(rank % world_size)
    
    # 载入数据
    dataset = TrainDataset(input=args.input, target=args.target)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset, batch_size=8, num_workers=1, drop_last=True,
        prefetch_factor=2, pin_memory=True, sampler=sampler
    )

    # 定义模型
    model = PETUNet().to(device)
    net = DDP(model)

    if args.resume and rank == 0:
        checkpoint_path = f"{args.output}/checkpoint/latest.pth"
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)

    # 优化器和调度器
    criterion = MSESSIMLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250], gamma=0.5)
    scaler = torch.amp.GradScaler("cuda")

    # 数据记录
    # writer = SummaryWriter(args.out_path+"%s"%args.task)
    
    # 初始化训练
    step = 0
    loss_all = np.zeros((300), dtype='float')
    num_batches = len(dataloader)
    
    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start to train")

    for epoch in range(EPOCHES):
        epoch_time = time.time()
        net.train()
        loss_this_time = 0
        dataloader.sampler.set_epoch(epoch)
        for _, sample_batched in enumerate(dataloader):
        
            input = sample_batched['input_img'].to(device, non_blocking=True)
            target = sample_batched['target_img'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                output_img = model(input)
                loss = criterion(output_img, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_this_time = loss_this_time + loss
            step += 1
            
        loss_this_time = loss_this_time / num_batches
        loss_all[epoch] = loss_this_time
            
        if epoch % 10 == 0 and rank == 0:
            state = net.state_dict()
            path1 = os.path.join(args.output, "checkpoint/%04d.pth"%epoch)
            torch.save(state, path1)
            shutil.copy2(path1, os.path.join(args.output, "checkpoint/latest.pth"))

        if rank == 0:
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch} | time: {(time.time() - epoch_time):.2f} | loss: {loss_this_time:.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")    
        
        scheduler.step()
    
    if rank == 0:
        state = net.state_dict()
        torch.save(state, os.path.join(args.output, "checkpoint/result.pth"))
        print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train finished.")
        
    cleanup()

if __name__ == '__main__':
    try:
        world_size = torch.cuda.device_count()
        args = parse_arguments()
        init_status(args)
        torch.set_num_threads(4)
        main(world_size, args)
    finally:
        cleanup()
