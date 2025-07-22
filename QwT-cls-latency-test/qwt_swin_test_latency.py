#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import random
import socket
import time
from functools import partial

import numpy as np
import pytorch_quantization.nn as quant_nn
import torch.distributed
import torch.nn as nn
import torch.utils.data
from timm.data.dataset import ImageDataset
from timm.models import create_model
from timm.utils import accuracy
from torch.utils.data import Dataset
from tqdm import tqdm

from models import quant_swin_transformer, swin_transformer

from utils.utils import write, create_transform, create_loader, AverageMeter, broadcast_tensor_from_main_process, \
    gather_tensor_from_multi_processes, compute_quantized_params, collect_stats_format, compute_amax_format

HOST_NAME = socket.getfqdn(socket.gethostname())

torch.backends.cudnn.benchmark = True
LINEAR_COMPENSATION_SAMPLES = 512

def onnx_export(q_model, dummy_input, suffix):
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(
        q_model,
        dummy_input,
        os.path.join(args.log_dir, '{}_bs_{}_{}.onnx'.format(args.model, dummy_input.size(0), suffix)),
        verbose=False,
        do_constant_folding=False
    )
    quant_nn.TensorQuantizer.use_fb_fake_quant = False

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class CompensationBlock(nn.Module):
    def __init__(self, W, b, r2_score, block, linear_init=True, local_rank=0, block_id=None):
        super(CompensationBlock, self).__init__()
        self.block = block

        self.lora_weight = nn.Parameter(torch.zeros((W.size(0), W.size(1))))
        self.lora_bias = nn.Parameter(torch.zeros(W.size(1)))

        if linear_init and (r2_score > 0):
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            if local_rank == 0:
                _write('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                _write('block {} using lora init'.format(block_id))

    def forward(self, x):
        out = self.block(x)
        if self.training:
            lora_weight = self.lora_weight.float()
            out = out + x @ lora_weight + self.lora_bias
        else:
            # QwT layers run in half mode
            lora_weight = self.lora_weight.half()
            out = out + (x.half() @ lora_weight).float() + self.lora_bias

        return out

def enable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_quant()


def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_quant()

class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]

def linear_regression(X, Y, block_id=0):
    X = X.reshape(-1, X.size(-1))

    X = gather_tensor_from_multi_processes(X, world_size=args.world_size)

    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))

    Y = gather_tensor_from_multi_processes(Y, world_size=args.world_size)

    _write('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))

    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    _write('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))

    return W, b, r2_score

@torch.no_grad()
def generate_compensation_model(q_model, train_loader, args):
    _write('start to generate compensation model')

    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, (image, _) in tqdm(enumerate(train_loader)):
        image = image.cuda()
        t_out = q_model.forward_before_blocks(image)
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()
        if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
            break


    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    output_previous = output_t
    for layer_id in range(len(q_model.layers)):
        current_layer = q_model.layers[layer_id]
        for block_id in range(len(current_layer.blocks)):

            feature_set.X = output_previous.detach().cpu()

            block = current_layer.blocks[block_id]
            output_full_precision = torch.zeros(size=[0, ], device=args.device)
            output_quant = torch.zeros(size=[0, ], device=args.device)
            output_t_ = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                disable_quant(block)
                full_precision_out = block(t_out)

                enable_quant(block)
                quant_out = block(t_out)

                output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
                output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
                output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size  // args.world_size - 1):
                    break

            assert torch.sum((output_previous - output_t_).abs()) < 1e-3
            global_block_id = sum(q_model.depths[:layer_id]) + block_id
            W, b, r2_score = linear_regression(output_t_, output_full_precision - output_quant, block_id=global_block_id)
            current_layer.blocks[block_id] = CompensationBlock(W=W, b=b, r2_score=r2_score, block=current_layer.blocks[block_id], linear_init=True if global_block_id >= args.start_block else False, local_rank=args.local_rank, block_id=global_block_id)
            q_model.cuda()

            QwT_block = current_layer.blocks[block_id]

            output_previous = torch.zeros(size=[0, ], device=args.device)
            for i, t_out in tqdm(enumerate(feature_loader)):
                t_out = t_out.cuda()
                enable_quant(QwT_block)
                previous_out = QwT_block(t_out)

                if (current_layer.downsample is not None) and (block_id == len(current_layer.blocks)-1):
                    previous_out = current_layer.downsample(previous_out)

                output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)

                torch.cuda.synchronize()
                if i >= (LINEAR_COMPENSATION_SAMPLES // args.batch_size // args.world_size - 1):
                    break

    return q_model


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="swin_tiny", choices=['swin_tiny', 'swin_small'], help="model")
parser.add_argument('--data_dir', default='/opt/Dataset/ImageNet', type=str)

parser.add_argument('--num_bits', default=8, type=int, help='bit-precision of weights')
parser.add_argument('--start_block', default=0, type=int)

parser.add_argument("--batch_size", default=32, type=int, help="batchsize of validation set")
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument("--seed", default=0, type=int, help="seed")
parser.add_argument('--save_files', default=False, action='store_true')

parser.add_argument("--local-rank", default=0, type=int)
args = parser.parse_args()

train_aug = 'large_scale_train'
test_aug = 'large_scale_test'
args.drop_path = 0.0
args.num_classes = 1000

model_type = args.model.split("_")[0]
if model_type == "deit":
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_pct = 0.875
elif model_type == 'vit':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    crop_pct = 0.9
elif model_type == 'swin':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_pct = 0.9
else:
    raise NotImplementedError

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
args.device = 'cuda:0'
args.world_size = 1
args.rank = 0  # global rank
if args.distributed:
    args.device = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.rank = torch.distributed.get_rank()

assert args.rank >= 0


args.log_dir = os.path.join('checkpoint', args.model, 'QwT', 'bs_{}_worldsize_{}_num_bits_{}_startblock_{}_sed_{}'.format(args.batch_size, args.world_size, args.num_bits, args.start_block, args.seed))

args.log_file = os.path.join(args.log_dir, 'log.txt')


if args.local_rank == 0:
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if os.path.isfile(args.log_file):
        os.remove(args.log_file)
else:
    time.sleep(1)

torch.cuda.synchronize()

_write = partial(write, log_file=args.log_file)

if args.distributed:
    _write('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
else:
    _write('Training with a single process on 1 GPUs.')
assert args.rank >= 0


def main():

    if args.local_rank == 0:
        _write(args)

    seed(args.seed)

    if args.local_rank == 0:
        _write('dataset mean : {} & std : {}'.format(mean, std))

    dataset_train = ImageDataset(root=os.path.join(args.data_dir, 'train'), transform=create_transform(train_aug, mean, std, crop_pct))
    dataset_eval = ImageDataset(root=os.path.join(args.data_dir, 'val'), transform=create_transform(test_aug, mean, std, crop_pct))

    if args.local_rank == 0:
        _write('len of train_set : {}    train_transform : {}'.format(len(dataset_train), dataset_train.transform))
        _write('len of eval_set : {}    eval_transform : {}'.format(len(dataset_eval), dataset_eval.transform))


    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=0.0,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=True,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    loader_eval = create_loader(
        dataset_eval,
        batch_size=args.batch_size,
        is_training=False,
        re_prob=0.,
        mean=mean,
        std=std,
        num_workers=args.num_workers,
        distributed=args.distributed,
        log_file=args.log_file,
        drop_last=False,
        local_rank=args.local_rank,
        persistent_workers=False
    )

    for data, target in loader_train:
        calib_data = data.to(args.device)
        break

    broadcast_tensor_from_main_process(calib_data, args)
    _write('local_rank : {} calib_data shape : {} value : {}'.format(args.local_rank, calib_data.size(), calib_data[0, 0, 0, :5]))

    base_model_zoo = {
        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
    }

    model_zoo = {
        'swin_tiny' : 'swin_tiny_patch4_window7_224_quant',
        'swin_small': 'swin_small_patch4_window7_224_quant',
    }

    base_model = create_model(base_model_zoo[args.model], num_classes=args.num_classes, pretrained=True)

    base_model.cuda()
    base_model.eval()

    q_model = create_model(model_zoo[args.model], num_classes=args.num_classes, pretrained=True, num_bits=args.num_bits, log_file=args.log_file, drop_path_rate=args.drop_path)

    q_model.cuda()
    q_model.eval()

    with torch.no_grad():
        assert calib_data.size(0) == 32
        collect_stats_format(q_model, calib_data)
        compute_amax_format(q_model, calib_method='percentile')

    q_model.cuda()

    base_params = compute_quantized_params(base_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('FP32 model size is {:.3f}'.format(base_params))
    top1_acc_eval = validate(base_model, loader_eval)
    _write('FP32   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    ptq_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('PTQ model size is {:.3f}'.format(ptq_params))
    top1_acc_eval = validate(q_model, loader_eval)
    _write('PTQ   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    if args.save_files and (args.local_rank == 0):
        if args.num_bits == 8:
            onnx_export(q_model,  torch.randn(64, 3, 224, 224, device='cuda'), 'ptq')
            onnx_export(base_model, torch.randn(64, 3, 224, 224, device='cuda'), 'fp32')
    torch.cuda.synchronize()

    q_model = generate_compensation_model(q_model, loader_train, args)

    if args.local_rank == 0:
        for n, m in q_model.named_modules():
            if isinstance(m, quant_nn.TensorQuantizer):
                _write('quant module : {}, calibrator : {}, enable_quant : {}, bits : {}'.format(n, m._calibrator, m._if_quant, m._num_bits))
                assert m._if_quant == True
                assert m._if_calib == False

    QwT_params = compute_quantized_params(q_model, local_rank=args.local_rank, log_file=args.log_file)
    _write('QwT model size is {:.3f}'.format(QwT_params))
    top1_acc_eval = validate(q_model, loader_eval)
    _write('QwT   eval_acc: {:.2f}'.format(top1_acc_eval.avg))

    if args.save_files and (args.local_rank == 0):
        if args.num_bits == 8:
            onnx_export(q_model,  torch.randn(64, 3, 224, 224, device='cuda'), 'QwT')
    torch.cuda.synchronize()


def validate(model, loader):
    top1_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):

            input = input.cuda()
            target = target.cuda()

            _, output = model(input)

            acc1, _ = accuracy(output, target, topk=(1, 5))

            top1_m.update(acc1.item(), output.size(0))

        top1_m.synchronize()

    _write('Test  Smples : {top1.count}    Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m))
    return top1_m


if __name__ == '__main__':
    main()
