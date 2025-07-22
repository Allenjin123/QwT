# Quantization without Tears

The following instructions explain how to reproduce QwT’s ImageNet classification results using the post-training quantization method **pytorch-percentile**. We also provide latency testing scripts to reproduce the results in Table 2 of [our paper](https://arxiv.org/abs/2411.13918).

### Environment Setup

To reproduce our results, set up your environment as follows:

#### Recommended dependencies
- timm 0.4.12  
- pytorch 2.4  
- pytorch_quantization 2.1.2 
- [nvidia‑tensorrt 10.4](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1040/index.html)


1. **Install CUDA 11.1**  
   - Download the CUDA 11.1 installer for Linux from NVIDIA:  
     ```bash
     wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
     sudo sh cuda_11.1.1_455.32.00_linux.run --silent --toolkit
     ```
   - Add CUDA to your `PATH` and `LD_LIBRARY_PATH` (e.g. in `~/.bashrc`):  
     ```bash
     export CUDA_HOME=/usr/local/cuda-11.1
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
     ```

2. **Install TensorRT 10.4.0 from TAR**  
   - Download the TensorRT 10.4.0 tar for your CUDA version from NVIDIA Developer (e.g. `TensorRT-10.4.0.XXXX.Ubuntu-18.04.cuda-11.1.tar.gz`).  
   - Choose an installation directory and set `TENSORRT_HOME`, for example:  
     ```bash
     export TENSORRT_HOME=/opt/TensorRT-10.4.0
     mkdir -p $TENSORRT_HOME
     ```  
   - Extract the tarball into that directory:  
     ```bash
     sudo tar -xzvf TensorRT-10.4.0.*.tar.gz -C $TENSORRT_HOME
     ```  
   - Update your environment variables:  
     ```bash
     export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
     export PATH=$TENSORRT_HOME/bin:$PATH
     export CPLUS_INCLUDE_PATH=$TENSORRT_HOME/include:$CPLUS_INCLUDE_PATH
     ```  

3. **Install the Python bindings and supporting packages**  
   ```bash
   pip install pycuda nvidia-pyindex
   
   # Adjust the wheel names / Python version suffix as needed
   pip install \
     $TENSORRT_HOME/python/tensorrt-*.whl \
   $TENSORRT_HOME/python/tensorrt_lean-*.whl \
   $TENSORRT_HOME/python/tensorrt_dispatch-*.whl \
    ```

## Evaluation

You can quantize and evaluate the model—including baseline **pytorch-percentile** and enhanced **pytorch-percentile + QwT**—using the same scripts:

```bash
# single GPU (ViT / DeiT)
python qwt_vit_and_deit_test_latency.py \
  --model [MODEL] \
  --data_dir [DATA_DIR] \
  --num_bits [W/A] \
  [--save_files]     # add this flag when using 8‑bit to export an ONNX model

# multi‑GPU (ViT / DeiT)
CUDA_VISIBLE_DEVICES=<GPUS> python -m torch.distributed.launch \
  --nproc_per_node=<N> --master_port=<PORT> \
  qwt_vit_and_deit_test_latency.py \
    --model [MODEL] \
    --data_dir [DATA_DIR] \
    --num_bits [W/A] \
    [--save_files]   # use with --num_bits 8 to dump ONNX for latency tests

# Swin
python qwt_swin_test_latency.py \
  --model [MODEL] \
  --data_dir [DATA_DIR] \
  --num_bits [W/A] \
  [--save_files]     # exports ONNX when quantizing to 8‑bit

```
`--save_files + --num_bits 8` will cause the script to export three ONNX files (batch size 64) into your log directory:

- \<model>_bs_64_fp32.onnx – FP32 baseline
- \<model>_bs_64_ptq.onnx – PTQ only   
- \<model>_bs_64_QwT.onnx – PTQ+QwT

You can then run `trtexec` on these three models to measure and compare latency.


## 🏎️ Latency Testing with TensorRT (`trtexec`)

After exporting your models to ONNX (e.g. via the `--save_files --num_bits 8` flag in the training scripts), you can build a TensorRT engine and measure end‑to‑end inference latency with `trtexec`. Below are example commands for the FP32 baseline, your PTQ model and your QwT‑enhanced model.

---

- **FP32 (no quantization)**  
  ```bash
  trtexec \
    --onnx=path/to/your/model_bs_64_fp32.onnx \
    --saveEngine=path/to/your/model_bs_64_fp32.trt
  ```
- **INT8 PTQ or PTQ + QwT**
  ```
  trtexec \
    --onnx=path/to/your/model_bs_64_ptq.onnx \
    --saveEngine=path/to/your/model_bs_64_ptq.trt \
    --best
   
   trtexec \
    --onnx=path/to/your/model_bs_64_qwt.onnx \
    --saveEngine=path/to/your/model_bs_64_qwt.trt \
    --best
  ```
- `--best` enables all supported precisions (FP32, FP16, INT8) and automatically selects the fastest kernels for optimal performance. Without this flag, only FP32 is activated.


After each run, the `trtexec` log will include detailed performance metrics (each prefixed by `[I]`), for example:

- **GPU Compute Time**: pure kernel execution time  
- **H2D/D2H Latency**: host↔device transfer times  
- **Enqueue Time**: overhead to enqueue each batch  
- **Total Host Walltime** and **Total GPU Compute Time**  

See [NVIDIA documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#using-the-trtexec-command-line-utility) for full details.

> **Note:**  
> We report **GPU Compute Time** as our latency metric because it isolates the pure device‑side compute cost of the network layers, excluding data transfers and host‑side overhead.  

### Evaluate your tensorrt engine
In addition to latency measurements, you can evaluate the ImageNet accuracy of your exported `.trt` engines using the provided `trt_evaluation.py` script. For example:

```bash
python trt_evaluation.py \
  --trt_path path/to/your/model_bs_64_qwt.trt \
  --data_dir /path/to/imagenet \
  --model_name deit_tiny
```
- --trt_path : Path to the TensorRT engine file (.trt) to evaluate (required)

- --data_dir : Root directory of the ImageNet dataset; must contain a val/ subfolder with class‐organized images

- --model_name : Model identifier for choosing preprocessing stats and transforms, e.g. deit_tiny, vit_base, swin_small

Note:
The script currently uses a fixed batch size of 64 and 4 DataLoader workers.

## Results on ImageNet 1K (W8/A8) and Latency @ BS=64

> **Hardware setup for latency measurements**  
> - CPU: Intel Xeon Gold 5220R @ 2.20 GHz (2 sockets, 24 cores/socket, 96 threads)  
> - GPU: NVIDIA GeForce RTX 3090 (24 GB)

| Model                     | Method                          | Top‑1 (%) W8/A8 | Latency (ms) @ BS 64 |
|:-------------------------:|:--------------------------------|:---------------:|:-------------------:|
| **DeiT‑T** (72.2)         | pytorch‑percentile              |       71.2         |         2.8           |
|                           | pytorch‑percentile + QwT        |       71.5         |         3.2           |
| **DeiT‑S** (79.9)         | pytorch‑percentile              |       74.9         |         6.0           |
|                           | pytorch‑percentile + QwT        |       78.9         |         6.7           |
| **DeiT‑B** (81.8)         | pytorch‑percentile              |       79.9         |         15.2           |
|                           | pytorch‑percentile + QwT        |       81.1         |         17.1           |
| **ViT‑S** (81.4)          | pytorch‑percentile              |       79.2         |         5.8           |
|                           | pytorch‑percentile + QwT        |       80.1         |         6.6           |
| **ViT‑B** (84.5)          | pytorch‑percentile              |       75.8         |         15.5           |
|                           | pytorch‑percentile + QwT        |       82.8         |         17.5           |
| **Swin‑T** (81.4)         | pytorch‑percentile              |       80.8         |         9.5           |
|                           | pytorch‑percentile + QwT        |       81.0         |         10.9           |
| **Swin‑S** (83.2)         | pytorch‑percentile              |       82.1         |         16.0           |
|                           | pytorch‑percentile + QwT        |       83.0         |         17.9           |

## Acknowledgements

This implementation leverages [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), the [PyTorch Quantization](https://docs.pytorch.org/docs/stable/quantization.html), and the [timm](https://github.com/rwightman/pytorch-image-models) library.


## Citation

We would greatly appreciate it if you could cite our paper if you find our implementation helpful in your work.

```bash
@InProceedings{Fu_2025_CVPR,
    author    = {Fu, Minghao and Yu, Hao and Shao, Jie and Zhou, Junjie and Zhu, Ke and Wu, Jianxin},
    title     = {Quantization without Tears},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4462-4472}
}
```