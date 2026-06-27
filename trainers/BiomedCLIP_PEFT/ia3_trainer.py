

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from .ia3_model import BiomedCLIP_IA3_Model


@TRAINER_REGISTRY.register()
class BiomedCLIP_IA3(TrainerX):


    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCLIP_IA3.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"\n>> IA3: {cfg.DATASET.NAME}  classes={classnames}")

        print(f"\nLoading BiomedCLIP...")
        biomedclip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        prec = cfg.TRAINER.BIOMEDCLIP_IA3.PREC
        if prec == "fp32" or prec == "amp":
            biomedclip_model = biomedclip_model.float()

        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        print("\nBuilding IA3 model...")
        self.model = BiomedCLIP_IA3_Model(
            cfg=cfg,
            classnames=classnames,
            biomedclip_model=biomedclip_model,
            tokenizer=tokenizer,
        )

        # Freeze: only IA3 wrapper is trainable
        print("\nFreezing BiomedCLIP (visual and text encoders)...")
        for name, param in self.model.named_parameters():
            if 'ia3' not in name:
                param.requires_grad_(False)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        ia3_params_count = sum(
            p.numel() for name, p in self.model.named_parameters() if 'ia3' in name
        )
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.4f}%)")
        print(f"  IA3 params: {ia3_params_count:,}")

        self.model.to(self.device)

        print(f"\nBuilding optimizer (LR={cfg.OPTIM.LR})...")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("ia3_model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"\nMultiple GPUs detected (n_gpus={device_count}), using DataParallel...")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.BIOMEDCLIP_IA3.PREC

        if prec == "amp":
            with autocast():
                outputs = model(image, label)
                loss = outputs["loss"]
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            outputs = model(image, label)
            loss = outputs["loss"]
            self.model_backward_and_update(loss)

        logits = outputs["logits"]
        acc = compute_accuracy(logits, label)[0].item()

        loss_summary = {"loss": loss.item(), "acc": acc}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        inp = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return inp, label

    def model_inference(self, input):
        return self.model(input, label=None)["logits"]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = f"{directory}/{name}/{model_file}"
            if not __import__('os').path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint.get('state_dict', checkpoint)
            epoch_val = checkpoint.get('epoch', epoch)

            new_state_dict = {
                k[7:] if k.startswith('module.') else k: v
                for k, v in state_dict.items()
            }

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch_val})')
            self._models[name].load_state_dict(new_state_dict, strict=False)
