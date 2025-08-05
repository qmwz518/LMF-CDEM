import argparse
import os
import math
from tkinter import Scale
import numpy as np
import random
import yaml
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from pytorch_msssim import ms_ssim, ssim

import datasets
import models
import utils
from test import eval_psnr
from utils import get_Device, plot_dem



def format_time(lasttime):
    # 提取小时和分钟
    hours = int(lasttime // 3600)  # 总秒数除以3600得到小时数
    minutes = int((lasttime % 3600) // 60)  # 剩余秒数除以60得到分钟数

    # 提取秒数并保留两位小数
    seconds = lasttime % 60  # 剩余秒数即为秒，保留两位小数

    # 格式化输出
    return f"{hours:02d}H,{minutes:02d}M,{seconds:05.2f}S"  # 秒数保留两位小数

class SIML1Loss(nn.Module):
    def __init__(
        self,
        alpha=0.84,     #diffloss和simloss的权重
        data_range=1.0,
        channel=1,
        ssim_mode="ssim",  # 新增模式选择参数
        win_size=7,
        weights=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.data_range = data_range
        self.channel = channel
        self.ssim_mode = ssim_mode.lower()  # 统一小写处理
        self.win_size = win_size
        self.weights = weights
        
        # 校验参数合法性
        assert self.ssim_mode in ["ssim", "ms-ssim"], "ssim_mode 必须为 'ssim' 或 'ms-ssim'"
        
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        # 根据模式选择结构相似性计算方式
        if self.ssim_mode == "ms-ssim":
            structural_similarity = ms_ssim(
                pred,
                target,
                data_range=self.data_range,
                win_size=self.win_size,
                size_average=True,
                weights=self.weights,
            )
        else:  # SSIM模式
            structural_similarity = ssim(
                pred,
                target,
                data_range=self.data_range,
                win_size=self.win_size,
                size_average=True,
            )

        # 计算结构相似性损失（1 - SSIM/MS-SSIM）
        structural_loss = 1 - structural_similarity
        
        # 计算L1损失
        l1_loss = self.l1_loss(pred, target)
        
        # 加权总损失
        total_loss = self.alpha *  l1_loss + (1 - self.alpha) * structural_loss
        
        return total_loss, structural_loss, l1_loss

# class MS_SSIM_L1_Loss(torch.nn.Module):
#     def __init__(self, alpha=0.84, data_range=1.0, channel=1):
#         super().__init__()
#         self.alpha = alpha       # MS-SSIM与L1的权重系数（参考论文常用值[3](@ref)）
#         self.data_range = data_range  # 数据范围（归一化后为1.0）
#         self.channel = channel  # DEM为单通道数据，设为1
#         self.l1_loss = torch.nn.L1Loss()

#     def forward(self, pred, target):
#         # 计算MS-SSIM损失（值越大越好，取负号转为损失）
#         ms_ssim_loss = 1 - ms_ssim(
#             pred, 
#             target, 
#             data_range=self.data_range, 
#             win_size=3, 
#             size_average=True,
#             weights=None,  # 默认多尺度权重
#         )
        
#         # 计算L1损失
#         l1_loss = self.l1_loss(pred, target)
        
#         # 加权组合损失[3,6](@ref)
#         total_loss = self.alpha * ms_ssim_loss + (1 - self.alpha) * l1_loss
        
#         return total_loss, ms_ssim_loss, l1_loss  # 返回总损失及各分项

class HybridLRScheduler:
    def __init__(self, optimizer, total_epochs, init_lr=0.001, T_mult=1):
        """
        混合学习率调度器
        :param optimizer: 优化器对象
        :param total_epochs: 总训练轮次
        :param init_lr: 初始学习率 (默认0.001)
        :param T_mult: Cosine退火周期倍增系数 (默认1)
        """
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # 第一阶段参数
        self.cosine_epochs = math.ceil(total_epochs * 0.5)  # 前50%使用Cosine
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=self.cosine_epochs,  # 余弦周期长度
            eta_min=init_lr*0.01       # 最小学习率为初始值的1%
        )
        
        # 第二阶段参数
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',        # 监控验证损失下降
            factor=0.5,        # 衰减系数
            patience=4,        # 容忍3个epoch不改善
            verbose=True
        )
        
    def step(self, val_loss=None):
        """
        更新学习率策略
        :param val_loss: 第二阶段需要的验证损失
        """
        self.current_epoch += 1
        
        if self.current_epoch <= self.cosine_epochs:
            # 第一阶段：使用余弦退火
            self.cosine_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
        else:
            # 第二阶段：使用高原衰减
            if val_loss is None:
                raise ValueError("第二阶段需要提供验证损失val_loss,如果在装载续训练，请从max_epoch/2开始训练")
            self.plateau_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {self.current_epoch}: ReduceLROnPlateau lr = {current_lr:.2e}')

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    # dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    trend = config.get('trend') # qumu
    coloss = config.get('coloss')
    bQBias = config.get('bQBias')
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset, 'trend':trend, 'coloss':coloss, 'bQBias':bQBias})


    log('{} dataset: size={} type={}'.format(tag, len(dataset),type(dataset[0])))
    for k, v in dataset[0][0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)if v is not None else None)) 
    

    sample_q = spec['wrapper']['args'].get('sample_q')
    print("train  sqmple_q=", sample_q)
    if tag == 'train': #by qumu
      config['sample_q_train'] = sample_q
    else: config['sample_q_val'] = sample_q
     
    bGPU = torch.cuda.is_available()
    nworkers = 4 if  bGPU else 0
    pin_memory = bGPU
    loader = DataLoader(dataset, batch_size=spec['batch_size'], shuffle=(tag == 'train'), num_workers=nworkers, pin_memory=pin_memory)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    device, bGPU = get_Device()
    if os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).to(device, non_blocking= bGPU)
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        print('='*70)
        print(f"torch.load(config['resume'] epoch_start={epoch_start}")
        lr_scheduType = config.get('schedule_lr_type')
        if  lr_scheduType is None:
            lr_scheduler = None
        else:
            if lr_scheduType == 'MultiStep':
                lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
                for _ in range(epoch_start - 1):
                    lr_scheduler.step()
            elif lr_scheduType == 'cosplateau':
                total_epochs=config['epoch_max']
                lr_scheduler = HybridLRScheduler(optimizer, total_epochs=total_epochs)
                
                for t in range(epoch_start - 1):
                    if t < total_epochs:
                        lr_scheduler.step()
                
    else:
        model = models.make(config['model'], args=dict(coloss=config.get('coloss'), bQBias = config.get('bQBias'))).to(device, non_blocking= bGPU)   #args=dict(coloss by qumu 331
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        lr_scheduType = config.get('schedule_lr_type')
        if  lr_scheduType is None:
            lr_scheduler = None
        else:
            if lr_scheduType == 'multistep':
                lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
            elif lr_scheduType == 'cosplateau':
                total_epochs=config['epoch_max']
                lr_scheduler = HybridLRScheduler(optimizer, total_epochs=total_epochs)
    
    
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, epoch): #utils.calc_psnr
    model.train()
    coloss = config.get('coloss') 
    btrend =  config.get('trend')
    bQBias = config.get('bQBias')  #预测趋势残差值 
    loss_type = None
    
    data_norm = config['data_norm']
    norm_on = data_norm['norm_on']
    scale = data_norm['model']['args']['max_scale']

    metric_fn = utils.calc_dem_metrics
    device, bGPU = get_Device()
    train_loss = utils.Averager()
    train_psnr = utils.Averager()
    train_msssim =  utils.Averager()

    if coloss:
        loss_type = config.get('loss_type')
        if loss_type == 'DEMGeoFeatsHybridLoss' and bQBias is False: # 当bQBias为True，预测DEM残差值，不使用DEMGeoFeatsHybridLoss
            loss_fn = utils.DEMGeoFeatsHybridLoss()
        else:
            loss_fn = SIML1Loss()
    else:
        loss_fn = nn.L1Loss()
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device, non_blocking=bGPU)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device, non_blocking=bGPU)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device, non_blocking=bGPU)
    gt_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device, non_blocking=bGPU)
    
    num_dataset = 800 # DIV2K
    
    iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
                        * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    for batch, batch_idx in tqdm(train_loader, leave=False, desc='train'):
        # 数据迁移至GPU
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=bGPU)
        
        # 归一化处理
       
        gt = batch['gt'] 
        if btrend == False:   #本身是0-1的值，如何norm_on为真，则再次归一化为-1~1间的值
          inp = (batch['inp'] - inp_sub) / inp_div if norm_on else batch['inp']
          pred = model(inp, batch['coord'], batch['cell'])
          pred_denorm = pred * gt_div + gt_sub if norm_on else pred #如何norm_on为真，将model输出从-1~1转换回0~1
        #   pred_denorm.clamp_(0, 1) # by qumu ???2025年7月9日
        else:
          inp = batch['inp']
          Trend = batch.get('DEMtrend')
          pred = model(inp, batch['coord'], batch['cell'], Trend)
          pred_denorm = (torch.tanh(pred) + 1) / 2 if not bQBias else pred # 0-1  
           
        # 损失计算
        #bQBias 时pred是预测的高程残差值
        if bQBias:
            hrBase = batch.get('hrBasedem')  #hrBasedem 低分辨率DEM细化采样得到
            # loss, ms_ssim_loss, l1_loss = loss_fn(pred, gt - hrBase)
            loss = loss_fn(pred, gt - hrBase)   # 预测的高程残差值，与低分辨率DEM细化采样得到的高程残差值的差值
        else:
            # All_loss = loss_fn(pred_denorm, gt)
            if coloss:
                All_loss = loss_fn(pred_denorm, batch)
                loss = All_loss[0]
                slopeloss = All_loss[2]
            else:
                loss = loss_fn(pred_denorm, gt)

        # 反向传播与参数更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        
        # 指标计算与记录
        # PSNR = metric_fn(pred, gt, OnlyPSNR=True)
        #这里如何不是coloss，则在训练阶段使用的是flatten的数据，则scale为None，否则为scale
        PSNR = utils.calc_psnr(pred_denorm, gt, scale=scale if coloss else None,data_range=1.0)
        writer.add_scalars('Loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        writer.add_scalars('PSNR', {'train': PSNR}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        train_psnr.add(PSNR)
        train_loss.add(loss.item())
        train_msssim.add(slopeloss) if coloss else 0

        # train_msssim.add(1 - ms_ssim_loss.item()) if coloss else 0
        # train_msssim.add(1 - ms_ssim_loss.item()) if coloss else 0

    if coloss:
      return train_loss.item(), train_psnr.item(),train_msssim.item() # 假设使用平均而非最后值
      
    else:
      return train_loss.item(), train_psnr.item() # 假设使用平均而非最后值


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None or config.get('data_norm')['norm_on'] == False:  #by qumu 
        # print('+'*40)
        # print('config.get(data_norm) is:',config.get('data_norm'))
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]},
            'norm_on': False
        }
    else:
      print(config.get('data_norm'))
      data_norm = config['data_norm']
  
    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    print('n_pgus=', n_gpus)
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)
    
    coloss = config['coloss']
    epoch_max = config['epoch_max']
    scheduler_lr_type = config.get('schedule_lr_type') #qumu 330
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    btrend =  config.get('trend')
    optim_sche = config.get('schedule_lr_type')
    local_ensemble = config['model']['args']['local_ensemble']   
    ensemAlpha = config['model']['args']['ensemAlpha'] 
    spatialatt = config['model']['args']['encoder_spec']['args']['spatialatt']
    n_feats = config['model']['args']['encoder_spec']['args']['n_feats']
    n_resblocks = config['model']['args']['encoder_spec']['args']['n_resblocks']
    conv_type = config['model']['args']['encoder_spec']['args']['conv_type']
    model_info = 'ensemble='+str(local_ensemble)+'_opti_='+ optim_sche+'_n_resblocks='+str(n_resblocks)+'_n_feats='+str(n_feats)+ '_btrend='+  str(btrend)+'_spatialatt='+str(spatialatt)+ '_conv_type='+conv_type
    print(model_info)

    seed = 10
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    # btrend =  config.get('trend')
    # bQBias = config.get('bQBias')  #预测趋势残差值 
    # spatialatt = config['model']['args']['encoder_spec']['args']['spatialatt']
    # conv_type = config['model']['args']['encoder_spec']['args']['conv_type']
    # print('btrend =',btrend,' spatialatt =',spatialatt, 'conv_type =',conv_type,' bQBias =',bQBias)
    day = datetime.datetime.now().strftime('%y%m%d%H')
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        traino = train(train_loader, model, optimizer, epoch)      
        if coloss:
            train_loss, train_psnr, train_mssim = traino[0],traino[1],traino[2]
        else: 
            train_loss, train_psnr = traino[0],traino[1]

        if lr_scheduler is not None:  #qumu 330
            #cosplateau # stepLR cosine plateau
            lr_scheduler.step(val_lss= train_loss) if  scheduler_lr_type == 'cosplateau' else  lr_scheduler.step() 

        log_info.append('train: loss={:.6f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        log_info.append('train: psnr={:.6f}'.format(train_psnr))
        writer.add_scalars('psnr', {'train': train_psnr}, epoch)
        if coloss:
        #   sim = 1-train_mssim
          log_info.append('train: train_mssim={:.6f}'.format(train_mssim))
          writer.add_scalars('mssim', {'train': train_mssim}, epoch)
        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        torch.save(sv_file, os.path.join(save_path, f'epoch-last{day}.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}-pyrenees.pth'.format(epoch)))
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            
            save_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evalrst'))
            save_folder = None # qumud
            val_res = eval_psnr(val_loader, model_,save_folder=save_folder,
                                data_norm=config['data_norm'],
                                eval_type=config.get('eval_type'),
                                eval_bsize=config.get('eval_bsize'),
                                coloss = coloss)
        
            
            
            if isinstance(val_res, tuple): #qumu
              log_info.append('val: psnr={:.4f}'.format(val_res[0]))
              log_info.append('val: ssim={:.4f}'.format(val_res[1]))
              writer.add_scalars('PSNR', {'val': val_res[0]}, epoch)
              writer.add_scalars('SSIM', {'val': val_res[1]}, epoch)
              val_res = val_res[0]
            else:
              log_info.append('val: psnr={:.4f}'.format(val_res))
              writer.add_scalars('PSNR', {'val': val_res}, epoch)
            
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-pyrenees-best320.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()
    log('Training Time:',format_time(timer.t()))
    torch.save(sv_file, os.path.join(save_path, f'epoch-last{day}.pth'))
    # torch.save(sv_file, os.path.join(save_path, 'epoch-pyrenees-best320.pth')) #qumud


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--savepath', default='./save')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(f'config loaded. savepath={args.savepath}')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join(args.savepath, save_name)
    
    main(config, save_path)
