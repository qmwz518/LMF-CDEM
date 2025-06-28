import os
import cv2
import json
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register

from .dem2be import *

import numpy as np
import matplotlib.pyplot as plt

def visualize_dem_hillshade(dem, hillshade, cmap='terrain', alpha=0.5, figsize=(12, 4)):
    """
    Visualize DEM and hillshade data side by side with a combined view.
    
    Args:
        dem: numpy array of elevation values
        hillshade: numpy array of hillshade values
        cmap: Colormap for DEM visualization
        alpha: Transparency for hillshade overlay
        figsize: Figure size (width, height)
    """
    # Print shapes for debugging
    print(f"DEM shape: {dem.shape}")
    print(f"Hillshade shape: {hillshade.shape}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot DEM
    dem_plot = ax1.imshow(dem, cmap=cmap)
    ax1.set_title('DEM')
    plt.colorbar(dem_plot, ax=ax1)
    
    # Plot hillshade
    ax2.imshow(hillshade, cmap='gray')
    ax2.set_title('Hillshade')
    
    # Plot combined view
    dem_plot = ax3.imshow(dem, cmap=cmap)
    ax3.imshow(hillshade, cmap='gray', alpha=alpha)
    ax3.set_title('Combined View')
    plt.colorbar(dem_plot, ax=ax3)
    
    plt.tight_layout()
    plt.show()


def To_tensor(elevation_data,transM ='origin'):
    """
    #by qumu
    Convert digital elevation data from a NumPy array to a PyTorch tensor.

    Parameters:
    elevation_data (np.ndarray): A 2D NumPy array representing digital elevation data.

    Returns:
    torch.Tensor: A PyTorch tensor with an added channel dimension.
    """
    # Ensure the input is a NumPy array
    if not isinstance(elevation_data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")

    # Convert the NumPy array to a PyTorch tensor
    # Add a channel dimension (H, W) -> (1, H, W)
    if transM != 'origin':
        tensor_data = torch.from_numpy(elevation_data).float()
        if transM != 'dem2multi':
            tensor_data = tensor_data.unsqueeze(0)
        else:
            tensor_data = tensor_data.permute(2, 0, 1)  #qumu from (H, W, C) to (C, H, W)
    else:
        tensor_data = transforms.ToTensor()(elevation_data.astype(np.float32))

    return tensor_data



# # 示例使用
# if __name__ == "__main__":
#     # 生成一个示例 DEM 数据 (低分辨率)
#     lr_dem = np.random.rand(64, 64).astype(np.float32) * 1000  # 假设高程范围为 [0, 1000]

#     # 提取趋势
#     trend = extract_trend_bilateral_filter(lr_dem, sigma_s=10, sigma_r=0.1)

#     # 计算残差
#     residual = lr_dem - trend


@register('dem-folder')
class DEMFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', 
                 bias_value=1e4, scale_value=10.0,
                 transM='origin',coloss=False):
        self.repeat = repeat
        self.cache = cache
        self.bias_value = bias_value
        self.scale_value = scale_value
        self.transM = transM
        self.coloss = coloss

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        print('root_path=',root_path, ' len(filenames):', len(filenames))        
        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'in_memory':
                dem_pkgs = self.dem2tensor(file)
                self.files.append(dem_pkgs)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            dem_pkgs = self.dem2tensor(x)
            if not self.coloss:
                print('return dem_pkgs')
                return dem_pkgs
            else:
                print('return (dem_pkgs, idx)')
                return (dem_pkgs, idx)

        elif self.cache == 'in_memory':
            if not self.coloss:
                return x
            else:
                return (x, idx)

    def dem2tensor(self,file):
        dem_data = demfile_io(file)
        # if self.coloss is not None: #qumu
        #     hillshade, slope, aspect = calculate_hillshade(dem_data) #25.1.5
        # # [scale, bias]
        add_args = [1.0, 0.0]
        if self.transM == 'origin':
            dem_data = dem_data
        elif self.transM == 'dem2one':
            dem_data, add_args = dem2one(dem_data=dem_data)
        elif self.transM == 'dem2one_tfasr':
            dem_data, add_args = dem2one(dem_data=dem_data, epsilon=10.0)
        elif self.transM == 'dem2multi':
            dem_data, add_args = dem2one(dem_data=dem_data)
            add_channels = dem2multi(dem_data=dem_data)
            dem_data = np.stack([dem_data, add_channels], axis=-1)
        else:
            raise Exception('Choose trans method.')
        
        return {
            'dem_data': To_tensor(dem_data,self.transM),
            'add_args': torch.tensor(add_args, dtype=torch.float32),
        }

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
