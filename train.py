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
from torchvision import transforms
from model.model import PETUNet
from model.loss import MSESSIMLoss

EPOCHS = 150
IMAGE_SIZE = 256
BATCH_SIZE = 8

class TrainDataset(Dataset):
    def __init__(self, input, target):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
        ])
        
        imput_img = np.array([input +"/"+ x  for x in os.listdir(input)])
        target_img = np.array([target +"/"+ x  for x in os.listdir(target)])
        
        assert len(imput_img) == len(target_img)
        
        imput_img.sort()
        target_img.sort()

        self.data = {'input': imput_img, 'target': target_img}
            
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

        if self.transform:  # 应用数据增强
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            torch.manual_seed(seed)  # 保证input和target应用相同的变换
            target_img = self.transform(target_img)
        
        sample = {
            'input_img': input_img,
            'target_img': target_img,
        }
        return sample

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../models/model_default", help="Path to save checkpoint.")
    parser.add_argument("--input", type=str, default="../mat/NAC_train_diffusion", help="Input images.")
    parser.add_argument("--target", type=str, default="../mat/CTAC_train_diffusion", help="Target images.")
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args()
    return args

def init_status(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.output+"/checkpoint", exist_ok=True)

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
        dataset, batch_size=BATCH_SIZE, sampler=sampler
    )
    dataset_val = TrainDataset(input="../mat/NAC_test", target="../mat/CTAC_test")
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    # 定义模型
    model = PETUNet().to(device)
    model = DDP(model)

    if args.resume and rank == 0:
        checkpoint_path = f"{args.output}/checkpoint/latest.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)

    # 优化器和调度器
    criterion = MSESSIMLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    
    # 初始化训练
    step = 0
    loss_all = np.zeros((EPOCHS), dtype='float')
    num_batches = len(dataloader)
    
    print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start to train")

    for epoch in range(EPOCHS):
        epoch_time = time.time()
        model.train()
        loss_this_time = 0
        dataloader.sampler.set_epoch(epoch)
        for _, sample_batched in enumerate(dataloader):

            input = sample_batched['input_img'].to(device, non_blocking=True)
            target = sample_batched['target_img'].to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda"):
                output_img = model(input)
                loss = criterion(output_img, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            loss_this_time = loss_this_time + loss
            step += 1
            
        loss_this_time = loss_this_time / num_batches
        loss_all[epoch] = loss_this_time
        
        if rank == 0:
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: {epoch + 1} | time: {(time.time() - epoch_time):.2f} | loss: {loss_this_time:.6f} | lr: {(optimizer.param_groups[0]['lr']):.6f}")    
            
        if (epoch + 1) % 10 == 0 and rank == 0:
            # 每十轮验证一次
            model.eval()
            loss_val = 0
            for batch in dataloader_val:
                input = batch['input_img'].to(device)
                target = batch['target_img'].to(device)
                with torch.no_grad():
                    output_img = model(input)
                loss = criterion(output_img, target)
                loss_val += loss
            loss_val_avg = loss_val / len(dataloader_val)
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Validation:: loss: {loss_val_avg}")
                
            state = model.state_dict()
            path1 = os.path.join(args.output, f"checkpoint/{epoch + 1}.pth")
            torch.save(state, path1)
            shutil.copy2(path1, os.path.join(args.output, "checkpoint/latest.pth"))
        
        scheduler.step()
    
    if rank == 0:
        state = model.state_dict()
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
