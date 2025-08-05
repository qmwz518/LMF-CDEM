"""
Modified from: https://github.com/yinboc/liif
"""

import os
import time
import shutil
import cv2

import numpy as np
np.sctypes = {
    'int': [np.int8, np.int16, np.int32, np.int64],
    'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
    'float': [np.float32, np.float64]
}

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torchvision import transforms
import imageio
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from tensorboardX import SummaryWriter

#by qumu
from pytorch_msssim import ms_ssim, ssim

import math

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamW':AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer



def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    version_float = float(''.join(torch.__version__.split('.')[:2]))
    if version_float>=18:
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    else:
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-value pairs.
        img: Tensor, (C, H, W)
    """
    coord = make_coord(img.shape[-2:])
    c=img.shape[0]
    img_value = img.view(c, -1).permute(1, 0)
    return coord, img_value

def resize_fn(img, size, method='nearest'):
    if method == 'bicubic':
        interpolation = transforms.InterpolationMode.BICUBIC
    elif method == 'bilinear':
        interpolation = transforms.InterpolationMode.BILINEAR
    elif method == 'nearest':
        interpolation = transforms.InterpolationMode.NEAREST
    else:
        raise Exception('Please align interpolation method')
    
    return transforms.Resize(size, interpolation)(img)


def Trend_of_DEM(IMG, Sigma=0.5):

    WinSize = (2 * np.ceil(2 * Sigma) + 1).astype(np.int32);
    Trend = cv2.GaussianBlur(IMG, (WinSize, WinSize), Sigma, borderType=cv2.BORDER_REPLICATE);
    return Trend



def normalize_data(data):
    """将数据归一化到 [0, 1] 范围"""
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min), data_min, data_max

def denormalize_data(data_normalized, data_min, data_max):
    """将数据反归一化到原始范围"""
    return data_normalized * (data_max - data_min) + data_min

def extract_trend_bilateral_filter(lr_dem, sigma_s=10, sigma_r=0.1):
    """
    使用双边滤波提取趋势（Trend）
    
    参数:
        lr_dem: 低分辨率 DEM 数据 (numpy 数组，形状为 H x W)
        sigma_s: 空间权重参数
        sigma_r: 强度权重参数（针对归一化数据）
    
    返回:
        trend: 提取的趋势 (numpy 数组，形状为 H x W)
    """
    # 归一化 DEM 数据
    lr_dem_normalized, data_min, data_max = normalize_data(lr_dem)
    
    # 将归一化数据转换为 8 位图像（OpenCV 双边滤波要求输入为 8 位）
    lr_dem_normalized_8u = (lr_dem_normalized * 255).astype(np.uint8)
    
    # 应用双边滤波
    trend_normalized_8u = cv2.bilateralFilter(lr_dem_normalized_8u, d=-1, sigmaColor=sigma_r*255, sigmaSpace=sigma_s)
    
    # 将滤波结果转换回浮点型并反归一化
    trend_normalized = trend_normalized_8u.astype(np.float32) / 255
    # trend = denormalize_data(trend_normalized, data_min, data_max)
    
    return trend_normalized

class DEMGeoFeatsHybridLoss(nn.Module):
    """
    综合高程L1损失、坡度损失、坡向损失和hillshade损失的混合损失函数。
    """
    def __init__(self, 
                 alpha=0.6,   # 高程L1损失权重
                 beta=0.2,    # 坡度损失权重
                 gamma=0.15,   # 坡向损失权重
                 delta=0.05,   # hillshade损失权重
                 data_range=1.0,
                 simloss = True,
                 eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.data_range = data_range
        self.eps = eps
        self.l1 = nn.L1Loss()

    def gradient(self, dem):
        # 计算x和y方向的梯度
        # dem: (B, 1, H, W)
        dzdx = dem[..., :, 2:] - dem[..., :, :-2]
        dzdx = dzdx / 2.0
        dzdx = torch.nn.functional.pad(dzdx, (1,1,0,0), mode='replicate')
        dzdy = dem[..., 2:, :] - dem[..., :-2, :]
        dzdy = dzdy / 2.0
        dzdy = torch.nn.functional.pad(dzdy, (0,0,1,1), mode='replicate')
        return dzdx, dzdy

    def compute_slope(self, dem):
        dzdx, dzdy = self.gradient(dem)
        slope = torch.atan(torch.sqrt(dzdx ** 2 + dzdy ** 2) + self.eps)
        return slope

    def compute_aspect(self, dem):
        dzdx, dzdy = self.gradient(dem)
        aspect = torch.atan2(dzdy, -dzdx + self.eps)
        # 将范围调整到0~2pi
        aspect = aspect % (2 * math.pi)
        return aspect

    def compute_geofeats_old(self, dem, azimuth=315, altitude=45):
        # azimuth: 光源方位角，单位度，0为北，顺时针
        # altitude: 光源高度角，单位度
        dzdx, dzdy = self.gradient(dem)
        # 转为弧度
        az = math.radians(azimuth)
        alt = math.radians(altitude)
        slope = torch.atan(torch.sqrt(dzdx ** 2 + dzdy ** 2) + self.eps)
        slope = slope % (0.5 * math.pi) / (0.5 * math.pi) #归一化到0~1
        aspect = torch.atan2(dzdy, -dzdx + self.eps)
        aspect = aspect % (2 * math.pi) 
        # hillshade公式
        hs = (torch.cos(alt) * torch.cos(slope) +
              torch.sin(alt) * torch.sin(slope) * torch.cos(az - aspect))
        # 归一化到0~1
        hs = (hs - hs.min()) / (hs.max() - hs.min() + self.eps)
        return slope, aspect, hs

    
    def compute_geofeats(self, dem, azimuth=315, altitude=45):
        device = dem.device  # 获取输入 DEM 的设备（CPU/GPU）
        dtype = dem.dtype    # 获取数据类型（如 float32）

        # 将角度转换为弧度，并创建为与 DEM 同设备的张量
        az = torch.tensor(math.radians(azimuth), device=device, dtype=dtype)
        alt = torch.tensor(math.radians(altitude), device=device, dtype=dtype)

        dzdx, dzdy = self.gradient(dem)
        slope = torch.atan(torch.sqrt(dzdx**2 + dzdy**2) + self.eps)
        slope = slope % (0.5 * math.pi) / (0.5 * math.pi)  # 归一化到 0~1
        aspect = torch.atan2(dzdy, -dzdx + self.eps)
        aspect = aspect % (2 * math.pi) 

        # 所有参数已转为 Tensor，可安全计算
        hs = (torch.cos(alt) * torch.cos(slope) +
            torch.sin(alt) * torch.sin(slope) * torch.cos(az - aspect))
        hs = (hs - hs.min()) / (hs.max() - hs.min() + self.eps)
        return slope, aspect, hs

    def forward(self, pred, target):
        # 高程L1损失
        gt = target['gt']
        gt_slope = target['slope']
        gt_aspect = target['aspect']
        gt_hs = target['hs']
        pred_slope, pred_aspect, pred_hs = self.compute_geofeats(pred)

        l1_loss = self.l1(pred, gt)
        # 坡度损失
        slope_loss = self.l1(pred_slope, gt_slope)
        # 坡向损失（用cos差异，避免角度环绕问题）
        # aspect_loss = self.l1(torch.cos(pred_aspect) - torch.cos(gt_aspect),
        #                       torch.sin(pred_aspect) - torch.sin(gt_aspect))
        aspect_cos_loss = self.l1(torch.cos(pred_aspect), torch.cos(gt_aspect))
        aspect_sin_loss = self.l1(torch.sin(pred_aspect), torch.sin(gt_aspect))
        aspect_loss = (aspect_cos_loss + aspect_sin_loss)/4.0
        
        # hillshade损失
        hillshade_loss = self.l1(pred_hs, gt_hs)

        total_loss = (self.alpha * l1_loss +
                      self.beta * slope_loss +
                      self.gamma * aspect_loss +
                      self.delta * hillshade_loss)
        return total_loss, l1_loss, slope_loss, aspect_loss, hillshade_loss


def calculate_hillshade(dem, azimuth=315, altitude=45):
    """
    Calculate hillshade from a DEM using only numpy arrays.
    
    Args:
        dem: numpy array of elevation values
        azimuth: Sun direction in degrees (default 315, NW)
        altitude: Sun elevation in degrees (default 45)
    
    Returns:
        numpy array of hillshade values (same size as input)
    """
    # Convert angles to radians
    azimuth = np.radians(azimuth)
    altitude = np.radians(altitude)
    
    # Calculate cell size (assuming square cells)
    x = 1.0
    y = 1.0
    
    # Pad DEM for gradient calculation
    dem_pad = np.pad(dem, pad_width=1, mode='edge')
    
    # Calculate gradients
    dz_dx = ((dem_pad[1:-1, 2:] - dem_pad[1:-1, :-2]) / (2 * x))
    dz_dy = ((dem_pad[2:, 1:-1] - dem_pad[:-2, 1:-1]) / (2 * y))
    
    # Calculate slope and aspect
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(dz_dy, -dz_dx)
    
    # Calculate hillshade
    hillshade = (np.cos(altitude) * np.cos(slope) + 
                 np.sin(altitude) * np.sin(slope) * 
                 np.cos(azimuth - aspect))
    
    # Scale to 0-255 range and clip
    hillshade = np.clip(hillshade * 255, 0, 255)
    
    # Ensure output has same shape as input by padding if necessary
    if hillshade.shape != dem.shape:
        diff_rows = dem.shape[0] - hillshade.shape[0]
        diff_cols = dem.shape[1] - hillshade.shape[1]
        if diff_rows > 0 or diff_cols > 0:
            hillshade = np.pad(hillshade, ((0, diff_rows), (0, diff_cols)), mode='edge')
    
    return hillshade, slope, aspect


'''close by qumu

def make_coord(shape, ranges=None, flatten=True):
    """
    Make coordinates at grid centers.
    """

    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """
    Convert the image to coord-RGB pairs.
    img: Tensor, (3, H, W)
    """

    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

'''


# def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
#     diff = (sr - hr) / rgb_range
#     if dataset is not None:
#         if dataset == 'benchmark':
#             shave = scale
#             if diff.size(1) > 1:
#                 gray_coeffs = [65.738, 129.057, 25.064]
#                 convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
#                 diff = diff.mul(convert).sum(dim=1)
#         elif dataset == 'div2k':
#             shave = scale + 6
#         else:
#             raise NotImplementedError
#         valid = diff[..., shave:-shave, shave:-shave]
#     else:
#         valid = diff[..., scale:-scale, scale:-scale]
#     mse = valid.pow(2).mean()
#     pnsr = -10 * torch.log10(mse)
#     return pnsr

def evaluate_model(model, dataloader, device,scale=1):
    model.eval()
    psnr_values, ssim_values, rmse_values = [], [], []
    with torch.no_grad():
        for lr, hr in dataloader:
            hr =  hr.to(device)
            reslr = lr[0].to(device)
            trendlr = lr[1].to(device)
            sr = model(reslr,trendlr)
            sr = sr.clamp(0, 1).cpu().numpy()
            hr = hr.cpu().numpy()
            for i in range(sr.shape[0]):
                psnr_values.append(psnr(hr[i, 0], sr[i, 0], data_range=1))
                ssim_values.append(ssim(hr[i, 0], sr[i, 0], data_range=1))
                rmse_values.append(np.sqrt(np.mean((hr[i, 0] - sr[i, 0]) ** 2)))
    return np.mean(psnr_values), np.mean(ssim_values), np.mean(rmse_values)

def get_Device():
  bGPU = torch.cuda.is_available()
  curdevice = torch.device("cuda") if bGPU else torch.device("cpu")
  return curdevice, bGPU
        
def calc_psnr(sr, hr, scale=None, data_range=1.0):
    # 计算sr和hr之间的均方误差
    diff = (sr - hr) / data_range
    # 如果scale为None，则valid为diff
    if scale is None:
        valid = diff
    # 否则，valid为diff去掉scale大小的边界
    else:
        shave = scale
        valid = diff[..., shave:-shave, shave:-shave]

    # 计算valid的平方和的平均值
    mse = valid.pow(2).mean()
    # 返回-10乘以mse的对数
    return -10 * torch.log10(mse)

def data2dem(data, file_pth, heatmap_flag=False):
    if isinstance(data, torch.Tensor):
        data=data.squeeze()
        data=(data-data.min())/(data.max()-data.min()+1.0e-6)

        if heatmap_flag:
            data = data.detach().cpu().numpy()
            heatmap_img = HeatmapsOnImage(data, shape=data.shape)
            imageio.imsave(file_pth,heatmap_img.draw(size=data.shape)[0])

        else:
            data=(data*255.0).floor()
            data = data.detach().cpu().numpy().astype(np.uint8)
            imageio.imsave(file_pth, data)

def data2heatmap(data, file_pth, max_value=1.0):
    data=data.squeeze()
    data = data.detach().cpu().numpy()
    heatmap_img = HeatmapsOnImage(data, shape=data.shape, max_value=max_value)
    imageio.imsave(file_pth,heatmap_img.draw(size=data.shape)[0])

def data2tif(data, file_pth):
    data=data.squeeze()
    data = data.detach().cpu().numpy()
    imageio.imsave(file_pth, data)


def plot_dem(dem_data, title='Digital Elevation Model'):
  """Plots a Digital Elevation Model (DEM) using matplotlib.

  Args:
    dem_data: A NumPy array containing the DEM data.
    title: The title of the plot (default: 'Digital Elevation Model').
  """
  plt.imshow(dem_data, cmap='terrain')
  plt.colorbar(label='Elevation')
  plt.title(title)
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()



#本函数只在coloss为True时使用，也即预测矩形DEM时使用，可只计算pnsr
def calc_dem_metrics(pred_tensor, true_tensor, data_range=None,OnlyPSNR=False,scale=None):
    """
    计算预测的超分辨率DEM与真实DEM之间的PSNR和SSIM指标。
    适用于PyTorch训练过程中的张量数据。
    
    参数:
        pred_tensor (torch.Tensor): 模型预测的DEM张量，形状为[B, 1, H, W]或[B, H, W]
        true_tensor (torch.Tensor): 真实DEM张量，形状为[B, 1, H, W]或[B, H, W]
        data_range (float, optional): 数据范围，用于PSNR和SSIM计算。如不提供，将自动计算。
       
        
    返回:
        dict: 包含PSNR和SSIM指标的字典
    """
    # # 确保输入是张量类型
    # if not isinstance(pred_tensor, torch.Tensor) or not isinstance(true_tensor, torch.Tensor):
    #     raise TypeError("输入必须是PyTorch张量")
    
    # # 确保张量在同一设备上
    # if pred_tensor.device != true_tensor.device:
    #     raise ValueError(f"预测张量和真实张量必须在同一设备上: {pred_tensor.device} vs {true_tensor.device}")
    
   
    # # 验证形状
    # if pred_tensor.shape != true_tensor.shape:
    #     raise ValueError(f"预测张量和真实张量形状必须一致: {pred_tensor.shape} vs {true_tensor.shape}")
    # # print(f'pred_tensor.shape={pred_tensor.shape}')
    # # pred_tensor.squeeze_(-1)
    # # true_tensor.squeeze_(-1)
    # # bs = pred_tensor.shape[0]
    # # if flattend:   #如果flattend为True，则将张量为展平状态
    # #     side_length  = int(math.sqrt(pred_tensor.shape[-1]))
    # #     pred_tensor = pred_tensor.view(bs,-1, side_length,side_length)
    # #     true_tensor = true_tensor.view(bs,-1, side_length,side_length)
    # # 如果未提供data_range，则自动计算
    if data_range is None:
        data_range = float(torch.max(true_tensor) - torch.min(true_tensor))
    
    # 计算PSNR
    # psnr_value = peak_signal_noise_ratio(pred_tensor, true_tensor, data_range=data_range)
    psnr_value = calc_psnr(pred_tensor, true_tensor, data_range=data_range,scale = scale)
    
    if OnlyPSNR:
      return psnr_value
    # 计算SSIM
    ssim_value = ssim(pred_tensor, true_tensor, data_range=data_range, win_size=5)
    ssim_value = ssim_value.mean().item()   
    return psnr_value,ssim_value

# 使用示例
if __name__ == "__main__":
    # 模拟批量训练数据
    batch_size = 4
    height, width = 64, 64
    
    # 创建示例张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_dem = torch.rand(batch_size, 1, height, width, device=device) * 1000  # 模拟DEM高程数据
    pred_dem = true_dem + torch.randn(batch_size, 1, height, width, device=device) * 50  # 添加噪声作为"预测"
    
    # 计算指标
    metrics = calculate_dem_metrics(pred_dem, true_dem)
    
    print(f"PSNR: {metrics['PSNR']:.4f} dB")
    print(f"SSIM: {metrics['SSIM']:.4f}")
    
    # 如果您想在训练循环中使用
    """
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 每N个批次计算并打印评估指标
            if batch_idx % log_interval == 0:
                metrics = calculate_dem_metrics(output, target)
                print(f"Epoch: {epoch}, Batch: {batch_idx}, PSNR: {metrics['PSNR']:.4f}, SSIM: {metrics['SSIM']:.4f}")
    """



