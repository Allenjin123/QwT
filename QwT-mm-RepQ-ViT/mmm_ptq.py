import torch


from models import load_model
from zero_shot import build_zero_shot_classifier, zero_shot
from utils.precision import get_autocast
from models.custom_model import ModelWithClassifier
from torch.utils.flop_counter import FlopCounterMode
from models.custom_model import replace_resblocks_with_transformer, replace_resblocks_with_vision_transformer
from termcolor import colored
from tqdm import tqdm
from torch.nn import Parameter
from quant.quant_model import quant_model, set_quant_state
from utils.precision import get_input_dtype
from dataset.dataset import get_wds_dataset
from quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul
from qwerty import generate_compensation_model, generate_compensation_model_2

class MMM_PTQ:
    def __init__(self, args):
        self.args = args
        self.model, self.preprocess, self.tokenizer = load_model(args)
        self.model = replace_resblocks_with_transformer(self.model)
        self.model = replace_resblocks_with_vision_transformer(self.model)
        self.model_to_device()
    
    def model_to_device(self):
        self.model.to(self.args.device)
        self.model.eval()
        
    def classifier_model_to_device(self):
        self.classifier_model.to(self.args.device)
        self.classifier_model.eval()
    
    def _create_classifier_model(self, data):
        autocast = get_autocast(self.args.precision)
        self.classifier_model = ModelWithClassifier(self.model, output_dim=len(data["imagenet-val"].classnames))
        self.classifier_model = build_zero_shot_classifier_with_linear_layer(self.classifier_model, self.tokenizer, data["imagenet-val"].classnames, data["imagenet-val"].templates, 20, self.args.device, True, autocast)
    
    def check_quantized(self):
        for n, m in self.classifier_model.named_modules():
            if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
                # print('quant module : {}, use_input_quant : {}, use_weight_quant : {}, bits : {} {}'.format(n, m.use_input_quant, m.use_weight_quant, m.weight_quantizer.n_bits, m.input_quantizer.n_bits))
                assert m.use_input_quant == True
                assert m.use_weight_quant == True
    
    
    def quantize_image_only_and_eval(self, data, qwerty=False):
        wq_params = {'n_bits': self.args.wq_params, 'channel_wise': True}
        aq_params = {'n_bits': self.args.aq_params, 'channel_wise': False}
        self._create_classifier_model(data)
        self.classifier_model = quant_model(self.classifier_model, input_quant_params=aq_params, weight_quant_params=wq_params)
        set_quant_state(self.classifier_model, input_quant=True, weight_quant=True)
        self.classifier_model_to_device()

        self.check_quantized()
        self.quant_ptq_1()
        if qwerty:
            self.qwerty()
        self.zero_shot_eval(data)
        
    
    def quantize_all_models_and_eval(self, data, qwerty=False):
        wq_params = {'n_bits': self.args.wq_params, 'channel_wise': True}
        aq_params = {'n_bits': self.args.aq_params, 'channel_wise': False}
        self.model = quant_model(self.model, input_quant_params=aq_params, weight_quant_params=wq_params)
        self.model_to_device()
        self.quant_ptq_2()

        if qwerty:
            self.qwerty_2()
        self._create_classifier_model(data)
        self.check_quantized()
        self.zero_shot_eval(data)
    
    def quant_ptq_1(self):
        input_dtype = get_input_dtype(self.args.precision)
        print(colored('Performing initial quantization...', 'cyan'))
        set_quant_state(self.classifier_model, input_quant=True, weight_quant=True)
        
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader

        for i, batch in tqdm(enumerate(calib_dataloader), total=self.args.iter, desc="Initial quantization"):
            if i >= self.args.iter:
                break
            images, texts = batch
            images = images.to(device=self.args.device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=self.args.device, non_blocking=True)
            calib_data = images, texts
            
            with torch.no_grad():
                images, texts = calib_data
                _ = self.classifier_model(images)
        
        # Scale reparameterization
        print(colored('Performing scale reparameterization ...', 'cyan'))
        scale_reparameterization(self.classifier_model.model.visual)
        
        # Re-calibration
        print(colored('Performing re-calibration ...', 'cyan'))
        set_quant_state(self.classifier_model, input_quant=True, weight_quant=True)
        
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader
        for i, batch in tqdm(enumerate(calib_dataloader), total=self.args.iter, desc="Re-calibration"):
            if i >= self.args.iter:
                break
            images, texts = batch
            images = images.to(device=self.args.device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=self.args.device, non_blocking=True)
            calib_data = images, texts
            
            with torch.no_grad():
                images, texts = calib_data
                _ = self.classifier_model(images)
                
    def quant_ptq_2(self):
        input_dtype = get_input_dtype(self.args.precision)
        print(colored('Performing initial quantization...', 'cyan'))
        set_quant_state(self.model, input_quant=True, weight_quant=True)
        
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader

        for i, batch in tqdm(enumerate(calib_dataloader), total=self.args.iter, desc="Initial quantization"):
            if i >= self.args.iter:
                break
            images, texts = batch
            images = images.to(device=self.args.device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=self.args.device, non_blocking=True)
            calib_data = images, texts
            
            with torch.no_grad():
                images, texts = calib_data
                _ = self.model(images, texts)
        
        # Scale reparameterization
        print(colored('Performing scale reparameterization ...', 'cyan'))
        scale_reparameterization(self.model.visual)
        scale_reparameterization(self.model)
        
        # Re-calibration
        print(colored('Performing re-calibration ...', 'cyan'))
        set_quant_state(self.model, input_quant=True, weight_quant=True)
        
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader
        for i, batch in tqdm(enumerate(calib_dataloader), total=self.args.iter, desc="Re-calibration"):
            if i >= self.args.iter:
                break
            images, texts = batch
            images = images.to(device=self.args.device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=self.args.device, non_blocking=True)
            calib_data = images, texts
            
            with torch.no_grad():
                images, texts = calib_data
                _ = self.model(images, texts)
    
    def zero_shot_eval(self, data):
        if not hasattr(self, 'classifier_model'):
            self._create_classifier_model(data)
        self.check_quantized()
        top1, top5 = zero_shot(self.classifier_model, data["imagenet-val"].dataloader, self.args)
        
        print(f"\033[1;32mTop-1 accuracy: {top1 * 100:.2f}\033[0m, \033[1;34mTop-5 accuracy: {top5 * 100:.2f}\033[0m")
        self.calculate_parameters(self.classifier_model)
        return top1, top5
    
    def qwerty(self):
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader
        self.classifier_model = generate_compensation_model(self.classifier_model, calib_dataloader, self.args)
        
    def qwerty_2(self):
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader
        self.model.visual = generate_compensation_model_2(self.model, self.model.visual, calib_dataloader, self.args)
        calib_dataloader = get_wds_dataset(self.args, self.preprocess, is_train=False, epoch=0, tokenizer=self.tokenizer).dataloader
        self.model = generate_compensation_model_2(self.model, self.model, calib_dataloader, self.args, if_image=False)
        
    
    def calculate_parameters(self, model, mid=False):
        original_params = 0
        for _name_, _module_ in model.named_modules():
            if len(_module_._parameters) > 0:
                for k in _module_._parameters:
                    if _module_._parameters[k] is not None:
                        if (k == 'weight') and hasattr(_module_, 'weight_quantizer'):
                            n_bits_ = _module_.weight_quantizer.n_bits
                        elif 'lora_weight' in k:
                            n_bits_ = 16
                        else:
                            n_bits_ = torch.finfo(_module_._parameters[k].dtype).bits

                        numel = _module_._parameters[k].numel()
                        num_bits = numel * n_bits_
                        original_params += num_bits
                        if mid:
                            print('original_params : {}.{} : {} * {} = {}'.format(_name_, k, n_bits_, numel, num_bits))
        print(f"\033[1;33mTotal parameters: {original_params // 8 / 1e6:.2f} MB\033[0m")
        return original_params // 8 / 1e6


# build zero-shot classifier with linear layer
def build_zero_shot_classifier_with_linear_layer(model, tokenizer, classnames, templates, num_classes_per_batch, device, use_tqdm, autocast):
    

    print(colored('Building zero-shot classifier...', 'cyan'))
    classifier_weights = build_zero_shot_classifier(
        model, tokenizer, classnames, templates, 
        num_classes_per_batch, device, 
        use_tqdm=True
    )
    # set new classifier weights
    with torch.no_grad():
        model.classifier.weight.copy_(classifier_weights.T)
        model.classifier.bias.copy_(torch.zeros_like(model.classifier.bias))
    
    return model


def scale_reparameterization(model):
    with torch.no_grad():
        module_dict={}
        q_model_slice = model.transformer.resblocks

        for name, module in tqdm(q_model_slice.named_modules(), desc="Processing modules"):
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if 'ln_1' in name or 'ln_2' in name or 'ln' in name:
                if 'ln_1' in name:
                    next_module = father_module.attn.qkv
                elif 'ln_2' in name:
                    next_module = father_module.mlp[0]
                else:
                    next_module = father_module.reduction
                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b

                
                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False