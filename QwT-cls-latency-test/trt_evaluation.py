import os
import torch
import argparse
import numpy as np
import tensorrt as trt
import torch.utils.data
import pycuda.autoinit
import pycuda.driver as cuda


from timm.data.dataset import ImageDataset
from tqdm import tqdm
from utils.utils import create_transform


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess_input(input_data, input_shape):
    input_data = input_data.astype(np.float32)
    input_data = input_data.reshape(input_shape)
    return input_data

def infer_new(engine, input_data):
    context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    inner_name = engine.get_tensor_name(1)
    output_name = engine.get_tensor_name(2)

    input_shape = context.get_tensor_shape(input_name)

    if -1 in input_shape:
        context.set_input_shape(input_name, input_data.shape)

    stream = cuda.Stream()

    inner_shape = context.get_tensor_shape(inner_name)
    inner_data = np.empty(inner_shape, dtype=np.float32)

    output_shape = context.get_tensor_shape(output_name)
    output_data = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_inner = cuda.mem_alloc(inner_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(inner_name, int(d_inner))
    context.set_tensor_address(output_name, int(d_output))

    cuda.memcpy_htod_async(d_input, input_data, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(inner_data, d_inner, stream)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    # stream.synchronize()

    return output_data


def evaluate(args):

    engine_path = args.trt_path
    model_name = args.model_name
    data_dir = args.data_dir

    test_aug = 'large_scale_test'
    model_type = model_name.split("_")[0]
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

    batch_size = 64

    dataset_eval = ImageDataset(root=os.path.join(data_dir, 'val'), transform=create_transform(test_aug, mean, std, crop_pct))
    test_loader = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    print('batch size : {}'.format(batch_size))
    print('dataset mean : {} & std : {}'.format(mean, std))
    print('len of eval_set : {}    eval_transform : {}'.format(len(dataset_eval), dataset_eval.transform))

    print('The evaluate model is {}'.format(engine_path))

    engine = load_engine(engine_path)
    total = 0
    correct = 0

    for tid, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
        input_data = preprocess_input(images.numpy(), engine.get_tensor_shape(engine.get_tensor_name(0)))
        output_data = infer_new(engine, input_data)
        predictions = np.argmax(output_data, axis=1)

        correct += (predictions == labels.numpy()).sum()
        total += labels.size(0)

        if tid % (len(test_loader) // 10) == 0:
            print(f"Batch {tid}, Accuracy so far: {correct / total:.4f}")

    # overall accuracy
    accuracy = correct / total
    print(f"Final Model Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trt_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='/opt/Dataset/ImageNet')
    parser.add_argument('--model_name', type=str, default='deit_tiny')

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()