import sys
import json
import torch
import warnings
warnings.filterwarnings("ignore")
from utils.params import parse_args
from utils.util import random_seed
from dataset.dataset import get_imagenet
from mmm_ptq import MMM_PTQ
from termcolor import colored
from dataset.dataset import get_wds_dataset


def main(args):
    # parse args
    args = parse_args(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.world_size = 1
    
    args_dict = vars(args)
    
    print(colored('Arguments:', 'cyan'), colored(json.dumps(args_dict, indent=4), 'yellow'))
    
    # set cudnn
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # set random seed
        random_seed(args.seed, 0)
        
    # load model
    mmm_ptq = MMM_PTQ(args)
    
    # load dataset, zero-shot so no need to load train data
    data = {}
    data["imagenet-val"] = get_imagenet(args, mmm_ptq.preprocess)

    if args.choice == 'fp32_eval':
        # run zero-shot evaluation for fp32  model
        mmm_ptq.zero_shot_eval(data)
    elif args.choice == 'image_only':
        # quantize only image encoder, args.qwerty means QwT flag
        mmm_ptq.quantize_image_only_and_eval(data, qwerty=args.qwerty)
    elif args.choice == 'all_quant':
        # quantize both image and text encoders, args.qwerty means QwT flag
        mmm_ptq.quantize_all_models_and_eval(data, qwerty=args.qwerty)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main(sys.argv[1:])