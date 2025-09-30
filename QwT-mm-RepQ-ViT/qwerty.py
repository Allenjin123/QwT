import warnings

import torch.utils
warnings.filterwarnings("ignore")
import torch.nn as nn
import random
import torch
import numpy as np

from utils.metadata import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


from quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'input_resolution')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', OPENAI_DATASET_MEAN)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', OPENAI_DATASET_STD)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


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
                print('block {} using linear init'.format(block_id))
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            if local_rank == 0:
                print('block {} using lora init'.format(block_id))

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
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            module.set_quant_state(input_quant=True, weight_quant=True)


def disable_quant(submodel):
    for name, module in submodel.named_modules():
        if isinstance(module, (QuantConv2d, QuantLinear, QuantMatMul)):
            module.set_quant_state(input_quant=False, weight_quant=False)

class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B

def linear_regression(X, Y, block_id=0):
    X = X.reshape(-1, X.size(-1))


    X_add_one = torch.cat([X, torch.ones(size=[X.size(0), ], device=X.device).reshape(-1, 1)], dim=-1)
    Y = Y.reshape(-1, Y.size(-1))


    print('the shape of X_add_one is {}, Y is {}'.format(X_add_one.size(), Y.size()))

    X_add_one_T = X_add_one.t()
    W_overall = torch.inverse(X_add_one_T @ X_add_one) @ X_add_one_T @ Y

    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b

    abs_loss = (Y - Y_pred).abs().mean()

    ss_tot = torch.sum((Y - Y.mean(dim=0)).pow(2))
    ss_res = torch.sum((Y - Y_pred).pow(2))
    r2_score = 1 - ss_res / ss_tot

    print('block : {}      abs : {:.6f}      r2 : {:.3f}'.format(block_id, abs_loss, r2_score))

    return W, b, r2_score

from torch.utils.data import Dataset
from tqdm import tqdm
class FeatureDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item]




@torch.no_grad()
def generate_compensation_model(model, train_loader, args):
    print('start generating compensation model')
    
    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, (image, _) in enumerate(train_loader):
        if i >= args.iter:
            break
        image = image.cuda()
        t_out = model.model.encode_image_forward_before_blocks(image)
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()
    
        

    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    output_previous = output_t
    
    for block_id in range(len(model.model.visual.transformer.resblocks)):
        
        feature_set.X = output_previous.detach().cpu()
        
        block = model.model.visual.transformer.resblocks[block_id]
        
        output_full_precision = torch.zeros(size=[0, ], device=args.device)
        output_quant = torch.zeros(size=[0, ], device=args.device)
        output_t_ = torch.zeros(size=[0, ], device=args.device)
        for i, t_out in tqdm(enumerate(feature_loader)):
            if i >= args.iter:
                break
            t_out = t_out.cuda()

            disable_quant(block)
            t_out = t_out.permute(1, 0, 2)
            full_precision_out = block(t_out)
            full_precision_out = full_precision_out.permute(1, 0, 2)
            enable_quant(block)
            quant_out = block(t_out)
            quant_out = quant_out.permute(1, 0, 2)

            t_out = t_out.permute(1, 0, 2)
            output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
            output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
            output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

            torch.cuda.synchronize()
        assert torch.sum((output_previous - output_t_).abs()) < 1e-3

        W, b, r2_score = linear_regression(output_t_, output_full_precision - output_quant, block_id=block_id)

        model.model.visual.transformer.resblocks[block_id] = CompensationBlock(W=W, b=b, r2_score=r2_score, block=model.model.visual.transformer.resblocks[block_id], linear_init=True if block_id >= 0 else False, local_rank=0, block_id=block_id)
        model.cuda()
        
        qwerty_block = model.model.visual.transformer.resblocks[block_id]

        output_previous = torch.zeros(size=[0, ], device=args.device)
        for i, t_out in tqdm(enumerate(feature_loader)):
            t_out = t_out.cuda()
            t_out = t_out.permute(1, 0, 2)  
            enable_quant(qwerty_block)
            previous_out = qwerty_block(t_out)
            previous_out = previous_out.permute(1, 0, 2)

            output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)

            torch.cuda.synchronize()
            if i >= args.iter - 1:
                break
    return model
    
    

@torch.no_grad()
def generate_compensation_model_2(base_model, model, train_loader, args, if_image=True):
    print('start generating compensation model')
    
    torch.cuda.synchronize()
    output_t = torch.zeros(size=[0,], device=args.device)
    for i, (image, text) in enumerate(train_loader):
        image = image.cuda()
        text = text.cuda()
        if if_image:
            t_out = base_model.encode_image_forward_before_blocks(image)
        else:
            t_out = base_model.encode_text_forward_before_blocks(text)
        print('first t_out shape is {}'.format(t_out.shape))
        output_t = torch.cat([output_t, t_out.detach()], dim=0)
        torch.cuda.synchronize()
    
        if i >= args.iter:
            break

    feature_set = FeatureDataset(output_t.detach().cpu())
    feature_loader = torch.utils.data.DataLoader(feature_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    output_previous = output_t
    
    for block_id in range(len(model.transformer.resblocks)):
        
        feature_set.X = output_previous.detach().cpu()
        
        block = model.transformer.resblocks[block_id]
        
        output_full_precision = torch.zeros(size=[0, ], device=args.device)
        output_quant = torch.zeros(size=[0, ], device=args.device)
        output_t_ = torch.zeros(size=[0, ], device=args.device)
        for i, t_out in tqdm(enumerate(feature_loader)):
            t_out = t_out.cuda()
            print('second t_out shape is {}'.format(t_out.shape))

            disable_quant(block)
            t_out = t_out.permute(1, 0, 2)
            full_precision_out = block(t_out)
            full_precision_out = full_precision_out.permute(1, 0, 2)

            enable_quant(block)
            quant_out = block(t_out)
            quant_out = quant_out.permute(1, 0, 2)

            t_out = t_out.permute(1, 0, 2)
            output_t_ = torch.cat([output_t_, t_out.detach()], dim=0)
            output_full_precision = torch.cat([output_full_precision, full_precision_out.detach()], dim=0)
            output_quant = torch.cat([output_quant, quant_out.detach()], dim=0)

            torch.cuda.synchronize()
            if i >= args.iter:
                break
        if block_id == 0:
            print('output_previous shape is {}'.format(output_previous.shape))
            print('output_t_ shape is {}'.format(output_t_.shape))
        assert torch.sum((output_previous - output_t_).abs()) < 1e-3

        W, b, r2_score = linear_regression(output_t_, output_full_precision - output_quant, block_id=block_id)

        model.transformer.resblocks[block_id] = CompensationBlock(W=W, b=b, r2_score=r2_score, block=model.transformer.resblocks[block_id], linear_init=True if block_id >= 0 else False, local_rank=0, block_id=block_id)
        model.cuda()
        
        qwerty_block = model.transformer.resblocks[block_id]

        output_previous = torch.zeros(size=[0, ], device=args.device)
        for i, t_out in tqdm(enumerate(feature_loader)):
            t_out = t_out.cuda()    
            t_out = t_out.permute(1, 0, 2)
            enable_quant(qwerty_block)
            previous_out = qwerty_block(t_out)
            previous_out = previous_out.permute(1, 0, 2)

            output_previous = torch.cat([output_previous, previous_out.detach()], dim=0)

            torch.cuda.synchronize()
            if i >= (1280 // args.batch_size):
                break
    return model