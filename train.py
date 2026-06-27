import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import torch
from yacs.config import CfgNode as CN

from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.utils import set_random_seed, setup_logger

# Register datasets used in the MSR manuscript.
import datasets.btmri
import datasets.busi
import datasets.chmnist
import datasets.covid
import datasets.ctkidney
import datasets.dermamnist
import datasets.kneexray
import datasets.kvasir
import datasets.lungcolon
import datasets.retina

# Register MSR and comparison trainers.
import trainers.BiomedCLIP_PEFT.linear_probe
import trainers.BiomedCLIP_PEFT.zeroshot_trainer
import trainers.BiomedCLIP_PEFT.vpt_trainer
import trainers.BiomedCLIP_PEFT.lora_trainer
import trainers.BiomedCLIP_PEFT.ia3_trainer
import trainers.BiomedCLIP_PEFT.adapter_trainer
import trainers.BiomedCLIP_PEFT.maple_trainer
import trainers.BiomedCLIP_PEFT.msr_trainer


def print_args(args, cfg):
    """Print command-line arguments and a concise config summary."""
    print("***************")
    print("** Arguments **")
    print("***************")
    for key in sorted(args.__dict__.keys()):
        print(f"{key}: {args.__dict__[key]}")

    print("\n*********************")
    print("** Config Summary **")
    print("*********************")

    from pprint import pformat

    cfg_summary = {
        "DATASET": {
            "NAME": cfg.DATASET.NAME,
            "ROOT": cfg.DATASET.ROOT,
            "NUM_SHOTS": cfg.DATASET.NUM_SHOTS,
            "SUBSAMPLE_CLASSES": cfg.DATASET.SUBSAMPLE_CLASSES,
        },
        "MODEL": {
            "BACKBONE": cfg.MODEL.BACKBONE.NAME,
        },
        "OPTIM": {
            "NAME": cfg.OPTIM.NAME,
            "LR": cfg.OPTIM.LR,
            "MAX_EPOCH": cfg.OPTIM.MAX_EPOCH,
            "LR_SCHEDULER": cfg.OPTIM.LR_SCHEDULER,
        },
        "TRAINER": cfg.TRAINER.NAME,
    }
    print(pformat(cfg_summary, indent=2, width=100))

    trainer_name = cfg.TRAINER.NAME.upper()
    trainer_specific = None
    if "ZEROSHOT" in trainer_name and hasattr(cfg.TRAINER, "BIOMEDCLIP_ZEROSHOT"):
        trainer_specific = {"BIOMEDCLIP_ZEROSHOT": dict(cfg.TRAINER.BIOMEDCLIP_ZEROSHOT)}
    elif "MSR" in trainer_name and hasattr(cfg.TRAINER, "MSR_BIOMEDCLIP"):
        trainer_specific = {"MSR_BIOMEDCLIP": dict(cfg.TRAINER.MSR_BIOMEDCLIP)}
    elif "MAPLE" in trainer_name and hasattr(cfg.TRAINER, "MAPLE_BIOMEDCLIP"):
        trainer_specific = {"MAPLE_BIOMEDCLIP": dict(cfg.TRAINER.MAPLE_BIOMEDCLIP)}
    elif "LORA" in trainer_name and hasattr(cfg.TRAINER, "BIOMEDCLIP_LORA"):
        trainer_specific = {"BIOMEDCLIP_LORA": dict(cfg.TRAINER.BIOMEDCLIP_LORA)}
    elif "IA3" in trainer_name and hasattr(cfg.TRAINER, "BIOMEDCLIP_IA3"):
        trainer_specific = {"BIOMEDCLIP_IA3": dict(cfg.TRAINER.BIOMEDCLIP_IA3)}
    elif "ADAPTER" in trainer_name and hasattr(cfg.TRAINER, "BIOMEDCLIP_ADAPTER"):
        trainer_specific = {"BIOMEDCLIP_ADAPTER": dict(cfg.TRAINER.BIOMEDCLIP_ADAPTER)}
    elif "VPT" in trainer_name and hasattr(cfg.TRAINER, "BIOMEDCLIP_VPT"):
        trainer_specific = {"BIOMEDCLIP_VPT": dict(cfg.TRAINER.BIOMEDCLIP_VPT)}
    elif trainer_name == "BIOMEDCLIP" and hasattr(cfg.TRAINER, "BIOMEDCLIP"):
        trainer_specific = {"BIOMEDCLIP": dict(cfg.TRAINER.BIOMEDCLIP)}

    if trainer_specific:
        print("\n*********************")
        print("** Trainer Config **")
        print("*********************")
        print(pformat(trainer_specific, indent=2, width=100))


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """Add config variables used by MSR and its comparison methods."""
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

    cfg.TRAINER.BIOMEDCLIP = CN()
    cfg.TRAINER.BIOMEDCLIP.PREC = "amp"

    cfg.TRAINER.BIOMEDCLIP_ZEROSHOT = CN()
    cfg.TRAINER.BIOMEDCLIP_ZEROSHOT.PREC = "amp"
    cfg.TRAINER.BIOMEDCLIP_ZEROSHOT.PROMPT_MODE = "custom"
    cfg.TRAINER.BIOMEDCLIP_ZEROSHOT.CUSTOM_TEMPLATE = ""

    cfg.TRAINER.BIOMEDCLIP_VPT = CN()
    cfg.TRAINER.BIOMEDCLIP_VPT.PREC = "amp"
    cfg.TRAINER.BIOMEDCLIP_VPT.VPT = CN()
    cfg.TRAINER.BIOMEDCLIP_VPT.VPT.N_PROMPTS = 20
    cfg.TRAINER.BIOMEDCLIP_VPT.VPT.LAYER_IDXS = [3, 6, 9]

    cfg.TRAINER.BIOMEDCLIP_LORA = CN()
    cfg.TRAINER.BIOMEDCLIP_LORA.PREC = "amp"
    cfg.TRAINER.BIOMEDCLIP_LORA.LORA = CN()
    cfg.TRAINER.BIOMEDCLIP_LORA.LORA.RANK = 4
    cfg.TRAINER.BIOMEDCLIP_LORA.LORA.ALPHA = 1.0
    cfg.TRAINER.BIOMEDCLIP_LORA.LORA.LAYER_IDXS = [3, 6, 9]
    cfg.TRAINER.BIOMEDCLIP_LORA.LORA.DROPOUT = 0.0

    cfg.TRAINER.BIOMEDCLIP_IA3 = CN()
    cfg.TRAINER.BIOMEDCLIP_IA3.PREC = "amp"
    cfg.TRAINER.BIOMEDCLIP_IA3.IA3 = CN()
    cfg.TRAINER.BIOMEDCLIP_IA3.IA3.LAYER_IDXS = [3, 6, 9]

    cfg.TRAINER.BIOMEDCLIP_ADAPTER = CN()
    cfg.TRAINER.BIOMEDCLIP_ADAPTER.PREC = "amp"
    cfg.TRAINER.BIOMEDCLIP_ADAPTER.ADAPTER = CN()
    cfg.TRAINER.BIOMEDCLIP_ADAPTER.ADAPTER.BOTTLENECK_DIM = 64
    cfg.TRAINER.BIOMEDCLIP_ADAPTER.ADAPTER.LAYER_IDXS = [3, 6, 9]
    cfg.TRAINER.BIOMEDCLIP_ADAPTER.ADAPTER.RESIDUAL_WEIGHT = 0.1

    cfg.TRAINER.MAPLE_BIOMEDCLIP = CN()
    cfg.TRAINER.MAPLE_BIOMEDCLIP.PREC = "amp"
    cfg.TRAINER.MAPLE_BIOMEDCLIP.VPT = CN()
    cfg.TRAINER.MAPLE_BIOMEDCLIP.VPT.N_PROMPTS = 5
    cfg.TRAINER.MAPLE_BIOMEDCLIP.VPT.LAYER_IDXS = list(range(12))
    cfg.TRAINER.MAPLE_BIOMEDCLIP.TEXT_PROMPT = CN()
    cfg.TRAINER.MAPLE_BIOMEDCLIP.TEXT_PROMPT.N_CTX = 4
    cfg.TRAINER.MAPLE_BIOMEDCLIP.TEXT_PROMPT.CTX_INIT = ""

    cfg.TRAINER.MSR_BIOMEDCLIP = CN()
    cfg.TRAINER.MSR_BIOMEDCLIP.PREC = "amp"
    cfg.TRAINER.MSR_BIOMEDCLIP.VPT = CN()
    cfg.TRAINER.MSR_BIOMEDCLIP.VPT.N_PROMPTS = 20
    cfg.TRAINER.MSR_BIOMEDCLIP.VPT.LAYER_IDXS = [3, 6, 9]
    cfg.TRAINER.MSR_BIOMEDCLIP.TEXT_PROMPT = CN()
    cfg.TRAINER.MSR_BIOMEDCLIP.TEXT_PROMPT.N_CTX = 4
    cfg.TRAINER.MSR_BIOMEDCLIP.TEXT_PROMPT.CTX_INIT = ""
    cfg.TRAINER.MSR_BIOMEDCLIP.SR_LAMBDA = 0.5
    cfg.TRAINER.MSR_BIOMEDCLIP.CONTRASTIVE_LAMBDA = 0.05
    cfg.TRAINER.MSR_BIOMEDCLIP.CONTRASTIVE_TEMP = 0.1
    cfg.TRAINER.MSR_BIOMEDCLIP.AGG_LAYER_IDXS = [3, 6, 9]
    cfg.TRAINER.MSR_BIOMEDCLIP.USE_COUPLING = True
    cfg.TRAINER.MSR_BIOMEDCLIP.USE_AGGREGATION = True
    cfg.TRAINER.MSR_BIOMEDCLIP.USE_CONTRASTIVE = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    reset_cfg(cfg, args)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg


def _format_seconds(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    return f"{secs:.1f}s"


def _to_plain_python(value):
    if isinstance(value, tuple):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, list):
        return [_to_plain_python(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def report_resource_usage(cfg, start_time):
    elapsed = time.perf_counter() - start_time
    resource = {
        "trainer": cfg.TRAINER.NAME,
        "dataset": cfg.DATASET.NAME,
        "seed": int(cfg.SEED),
        "batch_size": int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
        "input_size": _to_plain_python(cfg.INPUT.SIZE),
        "max_epoch": int(cfg.OPTIM.MAX_EPOCH),
        "total_train_time_sec": elapsed,
        "total_train_time_human": _format_seconds(elapsed),
    }

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.cuda.synchronize()
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        resource.update({
            "gpu_name": props.name,
            "gpu_total_memory_gb": props.total_memory / 1024**3,
            "peak_allocated_memory_gb": torch.cuda.max_memory_allocated(device_idx) / 1024**3,
            "peak_reserved_memory_gb": torch.cuda.max_memory_reserved(device_idx) / 1024**3,
        })

    print("\n**********************")
    print("** Resource Usage **")
    print("**********************")
    print(
        "Total training wall-clock time: "
        f"{resource['total_train_time_human']} ({resource['total_train_time_sec']:.2f} s)"
    )
    if "gpu_name" in resource:
        print(f"GPU: {resource['gpu_name']} ({resource['gpu_total_memory_gb']:.2f} GB)")
        print(f"Peak allocated GPU memory: {resource['peak_allocated_memory_gb']:.2f} GB")
        print(f"Peak reserved GPU memory: {resource['peak_reserved_memory_gb']:.2f} GB")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(cfg.OUTPUT_DIR, "resource_usage.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(resource, f, indent=2, ensure_ascii=False)
    print(f"Resource usage saved to: {save_path}")


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print(f"Setting fixed seed: {cfg.SEED}")
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    trainer = build_trainer(cfg)
    print("Trainer built successfully.")

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        if torch.cuda.is_available() and cfg.USE_CUDA:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        train_start_time = time.perf_counter()
        try:
            trainer.train()
        finally:
            report_resource_usage(cfg, train_start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", help="path to dataset root")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory for resume")
    parser.add_argument("--seed", type=int, default=-1, help="positive value enables fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/trainers/MSR_BiomedCLIP/btmri/btmri.yaml",
        help="path to method config file",
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="configs/datasets/btmri.yaml",
        help="path to dataset config file",
    )
    parser.add_argument("--trainer", type=str, default="MSR_BiomedCLIP", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    main(parser.parse_args())
