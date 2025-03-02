import argparse
import os, time
import json
import shutil
from pydicom import Dataset
from scipy import io
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

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

class PETUNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], in_channels=1, out_channels=1):
        super(PETUNet, self).__init__()
        
        # 下采样部分 (in_channels 为 1)
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU()
                )
            )
            in_channels = feature
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 上采样部分
        self.ups = nn.ModuleList()
        prev_feature = features[-1]
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(prev_feature + feature, feature, kernel_size=2, stride=2),
                    nn.BatchNorm2d(feature),
                    nn.ReLU()
                )
            )
            prev_feature = feature
        
        # 最终输出单通道 (out_channels = 1)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        
        # 下采样
        for down in self.downs:
            x = down(x)
            skips.append(x)
        
        x = self.bottleneck(x)
        
        # 上采样
        for up in self.ups:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        
        # 最后输出 (保持单通道)
        x = self.final_conv(x)
        return x
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../models/model_default", help="Path to save checkpoint.")
    parser.add_argument("--input", type=str, default="../mat/NAC_train", help="Input images.")
    parser.add_argument("--target", type=str, default="../mat/CT_train", help="Target images.")
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
        dataset, batch_size=16, num_workers=1, drop_last=True,
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
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250], gamma=0.5)

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
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time()
            
            input = sample_batched['input_img'].to(device, non_blocking=True)
            target = sample_batched['target_img'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            output_img = model(input)
            loss = criterion(output_img, target)
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                print('epoch: ' + str(epoch) + ' iter: ' + str(i_batch) +' loss: ' + str(loss.item()))
            
            loss_this_time = loss_this_time + loss
            step += 1
            
        loss_this_time = loss_this_time / num_batches
        loss_all[epoch] = loss_this_time
            
        if epoch % 10 == 0 and rank == 0:
            state = net.state_dict()
            path1 = os.path.join(args.output, "checkpoint/%04d.pth"%epoch)
            torch.save(state, path1)
            shutil.copy2(path1, os.path.join(args.output, "checkpoint/latest.pth"))
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch: %d Step: %d || loss: %.5f || lr: %f time: %f"%(
                epoch, step, loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time() - step_time
            ))

        scheduler.step()
        
        if rank == 0:
            print(f"[INFO {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch time: ", time.time()-epoch_time)    
    
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
