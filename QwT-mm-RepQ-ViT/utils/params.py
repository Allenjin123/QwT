import argparse
import ast


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--choice',
        type=str,
        required=True,
        help='flag for quantization choice',
        choices=['fp32_eval', 'image_only', 'all_quant']
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="webdataset",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='openai',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per GPU."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Default random seed."
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--wq_params",
        type=int,
        default=8,
        help="Weight quantization parameters.",
    )
    parser.add_argument(
        "--aq_params",
        type=int,
        default=8,
        help="Activation quantization parameters.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=4,
        help="Number of iterations.",
    )
    parser.add_argument(
        "--qwerty",
        action='store_true',
        help="Flag to enable qwerty functionality.",
    )
    args = parser.parse_args(args)


    return args
