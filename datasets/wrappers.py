import random
import math

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from datasets import register
import utils
from utils import to_pixel_samples
from utils import make_coord, resize_fn,Trend_of_DEM,calculate_hillshade


@register('dem-implicit-downsampled')
class DEMImplicitDownsampled(Dataset):
    def __init__(self,
                 dataset,
                 inp_size=None,
                 scale_min=1, 
                 scale_max=None,
                 sample_q=None,
                 coloss = False,
                 trend = False,
                 augment = True,
                 bQBias = False,
                 **kwargs,
                 ) -> None:
        '''
        Distribute the elevation value at (x,y)
        '''
        super().__init__()
        self.dataset = dataset
        self.length = len(dataset)

        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.coloss = coloss
        self.trend = trend
        self.bQBias = bQBias
        self.augment = augment
        self.hillshaecalc = utils.DEMGeoFeatsHybridLoss()


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dem_pkgs = self.dataset[idx]
        img = dem_pkgs['dem_data']
        add_args = dem_pkgs['add_args']

       
        s = random.uniform(self.scale_min, self.scale_max)
        bflatten = not self.coloss  #
        img_coord = make_coord(img.shape[-2:], flatten=bflatten)
        if self.coloss and self.sample_q is not None: #qumu
            img_coord = img_coord
            if self.inp_size is None:
                h_lr = math.floor(img.shape[-2] / s + 1e-9)
                w_lr = math.floor(img.shape[-1] / s + 1e-9)
                img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
                global_coord = img_coord[:round(h_lr * s), :round(w_lr * s),:]
                img_down = resize_fn(img, (h_lr, w_lr))
                crop_lr, crop_hr = img_down, img
            else:
                w_lr = self.inp_size
                w_hr = round(w_lr * s)
                # print('**********w_hr = round(w_lr * s)=',w_hr)
                X0 = random.randint(0, img.shape[-2] - w_hr)
                Y0 = random.randint(0, img.shape[-1] - w_hr)
                crop_hr = img[:, X0: X0 + w_hr, Y0: Y0 + w_hr]
                crop_lr = resize_fn(crop_hr, w_lr)                
                global_coord = img_coord[X0: X0 + w_hr, Y0: Y0 + w_hr,:]

            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                # dflip = random.random() < 0.5

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    return x
                crop_hr = augment(crop_hr)
                crop_lr = augment(crop_lr)
                

            # hillshade, slope, aspect = calculate_hillshade(crop_hr) 
            slope, aspect, hillshade = self.hillshaecalc.compute_geofeats(crop_hr)

            hr_coord = make_coord(crop_hr.shape[-2:],flatten=bflatten)
              # print('100 trenddem.shape:',trenddem.shape)
            x0 = random.randint(0, crop_hr.shape[-2] - self.sample_q) #这里的缩放比例s使用其不取LR图的边缘一像素位置
            y0 = random.randint(0, crop_hr.shape[-1] - self.sample_q)
            x1 = x0+self.sample_q #此时sample_q为LR边长值
            y1 = y0+self.sample_q
            hr_coord = hr_coord[x0:x1,y0:y1,:]  #[H,W,2]
            hr_value = crop_hr[:, x0:x1,y0:y1]
            hillshade = hillshade[:, x0:x1,y0:y1]
            slope = slope[:, x0:x1,y0:y1]
            aspect = aspect[:, x0:x1,y0:y1]
            global_coord = global_coord[x0:x1,y0:y1,:]
            #25.1.5
            if self.trend:
              trenddem = torch.tensor(Trend_of_DEM(crop_lr.numpy()))
            if self.bQBias:
              if self.trend:
                hrBasedem = F.grid_sample(trenddem.unsqueeze(0), hr_coord.unsqueeze(0),mode='nearest', align_corners=False,padding_mode='border')
                hrBasedem = hrBasedem.squeeze(0)
              else:
                hrBasedem = F.grid_sample(crop_lr.unsqueeze(0), hr_coord.unsqueeze(0),mode='nearest', align_corners=False,padding_mode='border')
                hrBasedem = hrBasedem.squeeze(0)
                # hrTrenddem = hrTrenddem[:, x0:x1,y0:y1] 
            cell = torch.ones_like(hr_coord.permute(2,0,1)) #[2,H,W]
            cell[0,...] *= 2 / crop_hr.shape[-2]  #会与inp进行.cat(),
            cell[1,...] *= 2 / crop_hr.shape[-1]
        else:   #不使用随机矩形范围值，flatternd
            img_coord, img_value = to_pixel_samples(img.contiguous())
            img_coord = img_coord.permute(1,0).view([-1, *(img.shape[1:])])
            if self.inp_size is None:
                h_lr = math.floor(img.shape[-2] / s + 1e-9)
                w_lr = math.floor(img.shape[-1] / s + 1e-9)
                img = img[:, :round(h_lr * s), :round(w_lr * s)] #assume round int
                img_down = resize_fn(img, (h_lr, w_lr))
                crop_lr, crop_hr = img_down, img
                global_coord = img_coord[:, :round(h_lr * s), :round(w_lr * s)]
            else:
                w_lr = self.inp_size
                w_hr = round(w_lr * s)
                X0 = random.randint(0, img.shape[-2] - w_hr)
                Y0 = random.randint(0, img.shape[-1] - w_hr)
                crop_hr = img[:, X0: X0 + w_hr, Y0: Y0 + w_hr]
                crop_lr = resize_fn(crop_hr, w_lr)
                #print('img_coord.shape=',img_coord.shape)
                global_coord = img_coord[:, X0: X0 + w_hr, Y0: Y0 + w_hr]

            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    return x
                crop_hr = augment(crop_hr)
                crop_lr = augment(crop_lr)
            global_coord = global_coord.reshape(2,-1).permute(1,0)
                        
            hr_coord, hr_value = to_pixel_samples(crop_hr.contiguous())
            if self.trend:
                trenddem = torch.tensor(Trend_of_DEM(crop_lr.numpy()))

            if self.sample_q is not None:  #  采样数统一为sample_q*sample_q
                sample_lst = np.random.choice(
                    len(hr_coord), self.sample_q**2, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_value = hr_value[sample_lst]
                global_coord = global_coord[sample_lst]
                
                # if self.trend == True and self.bQBias:
                #   hrTrenddem = F.grid_sample(trenddem.unsqueeze(0), hr_coord.unsqueeze(0).unsqueeze(0).flip(-1),mode='nearest', align_corners=False, padding_mode = 'border')[:, :, 0, :].permute(0, 2, 1)
                  
                #   hrTrenddem = hrTrenddem.squeeze(0)
                
                if self.bQBias:
                    if self.trend:
                        hrBasedem = F.grid_sample(trenddem.unsqueeze(0), hr_coord.unsqueeze(0),mode='nearest', align_corners=False,padding_mode='border')
                        hrBasedem = hrBasedem.squeeze(0)
                    else:
                        hrBasedem = F.grid_sample(crop_lr.unsqueeze(0), hr_coord.unsqueeze(0),mode='nearest', align_corners=False,padding_mode='border')
                        hrBasedem = hrBasedem.squeeze(0)                
                  # print(len(sample_lst))

                cell = torch.ones_like(hr_coord)
                cell[:, 0] *= 2 / crop_hr.shape[-2]
                cell[:, 1] *= 2 / crop_hr.shape[-1]


        # print(f'wrapperinp.shape:{crop_lr.shape} gt.shape{hr_value.shape}, hr_coord.shape:{hr_coord.shape}')
        if self.coloss == False:
            results = {
                'inp': crop_lr,
                'coord': hr_coord,
                'cell': cell,
                'gt': hr_value,
                'add_args': add_args,
                'global_coord': global_coord,
            }
        else:
            results = {
                'inp': crop_lr,
                'coord': hr_coord,
                'cell': cell,
                'gt': hr_value,
                'add_args': add_args,
                'global_coord': global_coord,
                'slope':slope,
                'aspect':aspect,
                'hillshade':hillshade,
            }
        # print('422wrapper, trend=',self.trend)
        if self.trend:
            results['DEMtrend'] = trenddem
            results['inp'] -= trenddem
        if  self.bQBias:
            results['hrBasedem'] = hrBasedem
        return results, idx


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    
    
# def resize_fn(img, size):
#     return transforms.ToTensor()(
#         transforms.Resize(size, InterpolationMode.BICUBIC)(
#             transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    
    




@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('sr-implicit-fixed-resolution')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_res, out_res):
        self.dataset = dataset
        self.inp_res = inp_res if inp_res else [720, 1280]
        self.out_res = out_res if out_res else [1440, 2560]
        self.aspect_ratio = inp_res[-1] / inp_res[-2]  # w / h

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        if self.aspect_ratio > img.shape[-1] / img.shape[-2]:
            # Crop to the aspect_ratio
            h_crop = round(img.shape[-1] / self.aspect_ratio)
            img = img[:, (img.shape[-2] - h_crop) // 2:(img.shape[-2] - h_crop) // 2 + h_crop, :]
        elif self.aspect_ratio < img.shape[-1] / img.shape[-2]:
            # Crop to the aspect_ratio
            w_crop = round(img.shape[-2] * self.aspect_ratio)
            img = img[:, :, (img.shape[-1] - w_crop) // 2:(img.shape[-1] - w_crop) // 2 + w_crop]

        [h_lr, w_lr] = self.inp_res
        [h_hr, w_hr] = self.out_res
        img_down = resize_fn(img, (h_lr, w_lr))

        coord = make_coord(self.out_res)
        inp_coord = make_coord(self.inp_res, flatten=False)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h_hr
        cell[:, 1] *= 2 / w_hr

        return {
            'inp': img_down,
            'inp_coord': inp_coord,
            'coord': coord,
            'cell': cell,
            # 'gt': hr_rgb
        }
    

#qumu
import torch
from torch.utils.data import DataLoader

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, max_scale, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.max_scale = max_scale  # 最大尺度因子

    def __iter__(self):
        for batch in super().__iter__():
            # 随机生成一个从 1 到 max_scale 之间的尺度因子
            scale_factor = torch.rand(1).item() * (self.max_scale - 1) + 1
            # 对批次中的每个样本应用相同的尺度因子
            scaled_batch = [(self.scale_sample(sample[0], scale_factor), sample[1]) for sample in batch]
            yield scaled_batch

    def scale_sample(self, sample, scale_factor):
        # 假设样本是一个单通道图像，使用插值进行缩放
        return torch.nn.functional.interpolate(
            sample.unsqueeze(0),  # 添加批次维度
            scale_factor=scale_factor,
            mode='bilinear',  # 使用双线性插值
            align_corners=False
        ).squeeze(0)  # 移除批次维度



@register('dem-downsampled-dataloader')
class DEMDataLoader(DataLoader):
    def __init__(self,
                 dataset,
                 inp_size=None,
                 scale_min = 1,
                 scale_max=None,
                 trend = False,
                 *args, 
                 **kwargs):
        super().__init__(dataset, *args, **kwargs)
        '''
        Distribute the elevation value at (x,y)
        '''
        self.dataset = dataset
        self.length = len(dataset)

        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.trend = trend


    def __len__(self):
        return self.length
    
    def __iter__(self):
        for batch in super().__iter__():
            # 随机生成一个从 1 到 max_scale 之间的尺度因子
             scale_factor = random.uniform(self.scale_min, self.scale_max)

            # 对批次中的每个样本应用相同的尺度因子
             scaled_batch = [self.scale_sample(sample, scale_factor) for sample in batch]
             yield scaled_batch

    def __getitem__(self, idx):
        dem_pkgs = self.dataset[idx]
        img = dem_pkgs['dem_data']
        # if self.coloss == True: qumu
        #     slope = dem_pkgs['slope']
        #     aspect = dem_pkgs['aspect']
        #     hillshade = dem_pkgs['hillshade']
        # print('*'*100)
        # print(img.shape)
        add_args = dem_pkgs['add_args']
        img_coord, img_value = to_pixel_samples(img.contiguous())
        img_coord = img_coord.permute(1,0).view([-1, *(img.shape[1:])])

        s = random.uniform(self.scale_min, self.scale_max)
        DEMtrend = None  #qumu
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
            # DEMtrend = torch.tensor(Trend_of_DEM(crop_lr.numpy()))
            
            # if self.coloss == True:
            #     slope = slope[:, :round(h_lr * s), :round(w_lr * s)]
            #     aspect = aspect[:, :round(h_lr * s), :round(w_lr * s)]
            #     hillshade = hillshade[:, :round(h_lr * s), :round(w_lr * s)] 


            global_coord = img_coord[:, :round(h_lr * s), :round(w_lr * s)]
            # crop_hr = img[:, :round(h_lr * s), :round(w_lr * s)]
            # crop_lr = resize_fn(crop_hr, (h_lr, w_lr))
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)
            global_coord = img_coord[:, x0: x0 + w_hr, y0: y0 + w_hr]
            # if self.coloss == True: #qumu
            #     slope = slope[:, x0: x0 + w_hr, y0: y0 + w_hr]
            #     aspect = aspect[:, x0: x0 + w_hr, y0: y0 + w_hr]
            #     hillshade = hillshade[:, x0: x0 + w_hr, y0: y0 + w_hr]
        
        if self.coloss == True:
          hillshade, slope, aspect = calculate_hillshade(crop_hr) #25.1.5
    
        global_coord = global_coord.reshape(2,-1).permute(1,0)
        # hr_value = crop_hr.reshape(img.shape[0],-1).permute(1,0)
 
        hr_coord, hr_value = to_pixel_samples(crop_hr.contiguous())
        
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_value = hr_value[sample_lst]
            global_coord = global_coord[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        # print(f'wrapperinp.shape:{crop_lr.shape} gt.shape{hr_value.shape}, hr_coord.shape:{hr_coord.shape}')
        if self.coloss == False:
            results = {
                'inp': crop_lr,
                'coord': hr_coord,
                'cell': cell,
                'gt': hr_value,
                'add_args': add_args,
                'global_coord': global_coord,
            }
        else:
            results = {
                'inp': crop_lr,
                'coord': hr_coord,
                'cell': cell,
                'gt': hr_value,
                'add_args': add_args,
                'global_coord': global_coord,
                'slope':slope,
                'aspect':aspect,
                'hillshade':hillshade,
            }
        # print('422wrapper, trend=',self.trend)
        if self.trend:
            trenddem = torch.tensor(Trend_of_DEM(crop_lr.numpy()))
            results['DEMtrend'] = trenddem
            results['inp'] -= trenddem
        return results, idx

'''
# 使用自定义 DataLoader
dataset = ...  # 你的数据集
max_scale = 2.0  # 设定最大尺度因子
train_loader = CustomDataLoader(dataset, max_scale, batch_size=32, shuffle=True, num_workers=4)
'''