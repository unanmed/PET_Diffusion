import argparse
import os
from scipy import io
import torch
import numpy as np
import cv2
from tqdm import tqdm
from model.model import PETUNet
from model.loss import MSESSIMLoss
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, generate_uid
from skimage import exposure

def parse_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    parser.add_argument("--output", type=str, default="../results/model_default", help="Path to save checkpoint.")
    parser.add_argument("--input", type=str, default="../mat/NAC_test", help="Input images.")
    parser.add_argument("--target", type=str, default="../mat/CTAC_test", help="Target images.")
    parser.add_argument("--model", type=str, default="../models/model_default/checkpoint/latest.pth")
    args = parser.parse_args()
    return args

def init_status(args):
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "predict"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "predict_dcm"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "predict_png"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "target"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "origin"), exist_ok=True)
    
def process_image(img):
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).unsqueeze(0).float().cuda()
    return img

def save_as_dicom(output, file_path):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()  # 使用 generate_uid() 生成唯一的UID
    file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1')  # Explicit VR Little Endian

    ds = FileDataset(file_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Anonymous"
    ds.PatientID = "123456"
    ds.Modality = "PET"
    ds.SeriesInstanceUID = generate_uid()  # 使用 generate_uid() 生成唯一的UID
    ds.StudyInstanceUID = generate_uid()  # 使用 generate_uid() 生成唯一的UID
    ds.FrameOfReferenceUID = generate_uid()  # 使用 generate_uid() 生成唯一的UID
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # 0表示无符号整数
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = output.shape
    ds.PixelData = output.tobytes()

    # 补充更多必要的标签
    ds.SOPInstanceUID = generate_uid()  # 使用 generate_uid() 生成唯一的UID
    ds.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.2')  # CT Image Storage
    ds.StudyDate = "20231001"
    ds.SeriesDate = "20231001"
    ds.ContentDate = "20231001"
    ds.StudyTime = "120000"
    ds.SeriesTime = "120000"
    ds.ContentTime = "120000"
    ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.RescaleType = "HU"

    # 设置字节顺序和 VR
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # 保存 DICOM 文件
    ds.save_as(file_path)

def eval(args):
    # 🧠 2. 加载训练好的模型权重
    model = PETUNet()
    checkpoint = torch.load(args.model, map_location='cuda')
    criterion = MSESSIMLoss()
    
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model.eval()  # 设为推理模式
    
    input_folder = args.input
    target_folder = args.target
    output_folder = args.output
    
    loss_total = 0
    
    for file_name in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file_name)
        target_path = os.path.join(target_folder, file_name)
        
        input_img = io.loadmat(input_path)['img'].astype('float32')
        input_img_tensor = process_image(input_img)
        
        target_img = io.loadmat(target_path)['img'].astype('float32')
        target_img_tensor = process_image(target_img)
        
        with torch.no_grad():
            output = model(input_img_tensor)
            
        loss = criterion(output, target_img_tensor)
        loss_total += loss.item()
        
        # 处理输出
        output = output.squeeze().cpu().numpy()
        
        # 归一化到 0-255（用于 PNG 保存）
        output_png = (output - output.min()) / (output.max() - output.min()) * 255
        output_png = output_png.astype(np.uint8)
        
        # 归一化到 0-65535（用于 DICOM 保存）
        # 将输出图像转换为CT图像的HU值范围
        output_dcm = (output - output.min()) / (output.max() - output.min()) * 4000 - 1000  # 假设范围为-1000到3000
        output_dcm = output_dcm.astype(np.int16)  # 使用int16类型存储HU值
                
        # 保存为DICOM文件
        dicom_output_path = os.path.join(output_folder, "predict_dcm", f"{os.path.splitext(file_name)[0]}.dcm")
        save_as_dicom(output_dcm, dicom_output_path)
     
        # 处理 input_img 和 target_img 以匹配输出
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min()) * 255
        input_img = input_img.astype(np.uint8)

        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min()) * 255
        target_img = target_img.astype(np.uint8)

        # 拼接三张图像
        combined_img = np.concatenate([input_img, output_png, target_img], axis=1)
        
        # 保存拼接后的图像
        output_path = os.path.join(output_folder, "predict_png", f"{os.path.splitext(file_name)[0]}.png")
        cv2.imwrite(output_path, combined_img)
        
    print(f"All processed loss: {loss_total / len(os.listdir(input_folder))}")
    
if __name__ == "__main__":
    args = parse_arguments()
    init_status(args)
    eval(args)