import argparse
import os
import math
from functools import partial

import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from utils import get_Device,plot_dem, calc_dem_metrics


def batched_predict(model, inp, coord, cell, bsize):
    """
    Perform batched predictions using a model on provided inputs.

    :param model: The model used for generating predictions.
    :param inp: Input data to the model. Typically, this is a batch of images or features.
    :param coord: Coordinates associated with the input data, used for models that require spatial context.
    :param cell: Scaling factors or additional data related to each coordinate, supporting the model's prediction.
    :param bsize: Batch size used to split the data into manageable chunks during prediction.
    :return: The concatenated tensor of predictions for all batches.
    """

    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4, fast=False,
              verbose=False, save_folder=None,btrend=False,coloss=False,channels=1):
    """
    Evaluate the Peak Signal-to-Noise Ratio (PSNR) of a model over a dataset loaded through a specified loader.

    :param loader: The DataLoader providing the input data.
    :param model: The model to be evaluated.
    :param data_norm: Normalization parameters for input and ground truth data.
    :param eval_type: Type of evaluation to perform, e.g., specific scales or datasets.
    :param eval_bsize: Batch size for evaluation, can be None for full-batch processing.
    :param window_size: The size of windowing for input padding, useful in SwinIR.
    :param scale_max: Maximum scale factor used in model training.
    :param fast: If True, evaluates using none-batched method.
    :param verbose: If True, provides detailed progress output.
    :param save_folder: Folder path where the resulting images are saved.
    :return: The average PSNR over all evaluated data.
    """

    model.eval()
    device, bGPU = get_Device()

    if data_norm is None or data_norm['norm_on'] == False:
        norm_on = False  #qumuu
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    else : norm_on = data_norm['norm_on']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device, non_blocking= bGPU)
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device, non_blocking= bGPU)
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device, non_blocking= bGPU)
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device, non_blocking= bGPU)
    # print(f'eval_type:  {eval_type}')
    if eval_type is None:
        scale = scale_max
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    if coloss:
      # metric_fn = partial(calc_dem_metrics, data_range=1)
      metric_fn = calc_dem_metrics

    # print('test82 metric_fn',metric_fn)
    val_res = utils.Averager()
    val_ssim = utils.Averager()
    index = 1
    pbar = tqdm(loader, leave=False, desc='val')
    for batch,batch_idx in pbar:
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking= bGPU)

        inp = (batch['inp'] - inp_sub) / inp_div if norm_on else batch['inp']
        # print(f'test 105 inp max={inp.min()} inp_div={inp_div}')
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad))).unsqueeze(0).to(device)
            print('105 window_size=',window_size)
            print('106 coord = utils.make_coord((scale*(h_old+h_pad) coord.shape=',coord.shape) # qumud
            cell = torch.ones_like(coord)
            cell[:, :, 0] *= 2 / inp.shape[-2] / scale
            cell[:, :, 1] *= 2 / inp.shape[-1] / scale
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']

        # Cell clip for extrapolation
        # print('++'*50)
        # print(f"batch_idx={batch_idx},len of batch['inp']:{len(batch['inp'])},len(coord):{len(coord)}")
        # print('--'*50)
        if eval_bsize is None:
            with torch.no_grad():
                # print('test130 eval_bsize is None:')
                pred = model(inp, coord, cell*max(scale/scale_max, 1), batch.get('DEMtrend'))
        else:
            if fast:
                print('test130  if fast::')``
                with torch.no_grad():
                    pred = model(inp, coord, cell*max(scale/scale_max, 1),batch.get('DEMtrend'))
            else:
                # print('test139 batched_predict:')
                pred = batched_predict(model, inp, coord, cell*max(scale/scale_max, 1), batch.get('DEMtrend'), eval_bsize)
        # print('pred.min()=',pred.min())   
        pred_denorm = pred * gt_div + gt_sub if norm_on else pred
        # print('test43 gt_div=',gt_div.numpy(),'pred.min()=',pred.min())
        pred_denorm.clamp_(0, 1)

        # save sr image
        save_folder = save_folder if save_folder is not None else eval_type
        if save_folder is not None:
            ih, iw = batch['inp'].shape[-2:]
            save_img = pred_denorm.view(round((ih + h_pad) * scale),
                                 round((iw + w_pad) * scale), channels).permute(2, 0, 1).cpu()
            save_path = "./outputs/" + save_folder
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            index_str = str(index) if index >= 100 else ('0' + str(index) if index >= 10 else '00' + str(index))
            transforms.ToPILImage()(save_img).save(save_path + "/" + save_folder + "_" + index_str + ".png")
            index += 1

        if eval_type is not None:  # reshape for shaving-eval
            # gt reshape
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            
            # prediction reshape
            ih += h_pad
            iw += w_pad
            s = math.sqrt(coord.shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred_denorm = pred_denorm.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            pred_denorm = pred_denorm[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
        
        #qumud320
        # predv = pred.cpu().numpy().squeeze()
        # gtv = batch['gt'].cpu().numpy().squeeze()
        # print(f'test170 pred.shape={pred.shape},predv.shape={predv.shape},gtv.shape={gtv.shape}')
        # print(predv[:3])
        # print('*'*100)
        # print(gtv[:3])


        res = metric_fn(pred_denorm, batch['gt'])
        # print(f"test183 pred.shape={pred.shape}")
        if isinstance(res, tuple): #qumu
          val_res.add(res[0], inp.shape[0])
          val_ssim.add(res[1], inp.shape[0])
        else:
          val_res.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description(f'val: {val_res.item():.4f} ssim: {val_ssim:.4f}')
            # pbar.set_description('val {:.4f}'.format(val_res.item()))
            

    psnr, ssim = val_res.item(), val_ssim.item()        
    return psnr, ssim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--fast', default=True)  # Set fast to True for LMF, False for original LIIF/LTE/CiaoSR
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--cmsr', default=False)
    parser.add_argument('--cmsr_mse', default=0.00002)
    parser.add_argument('--cmsr_path')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'], num_workers=8, pin_memory=True)

    if args.cmsr:
        try:
            # Test with CMSR
            with open(args.cmsr_path, 'r') as f:
                s2m_tables = yaml.load(f, Loader=yaml.FullLoader)
            cmsr_spec = {
                "mse_threshold": float(args.cmsr_mse),
                "path": args.cmsr_path,
                "s2m_tables": s2m_tables,
                "log": False,
            }
        except FileNotFoundError:
            cmsr_spec = None
    else:
        cmsr_spec = None

    model_spec = torch.load(args.model)['model']
    model_spec["args"]["cmsr_spec"] = cmsr_spec
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader,
                    model,
                    data_norm=config.get('data_norm'),
                    eval_type=config.get('eval_type'),
                    eval_bsize=config.get('eval_bsize'),
                    window_size=int(args.window),
                    scale_max=int(args.scale_max),
                    fast=args.fast,
                    verbose=True)
    print('result: {:.4f}'.format(res))
