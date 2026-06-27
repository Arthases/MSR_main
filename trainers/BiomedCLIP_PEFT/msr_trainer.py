import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer

from .msr_model import MSR_BiomedCLIP_Model


@TRAINER_REGISTRY.register()
class MSR_BiomedCLIP(TrainerX):
   

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MSR_BIOMEDCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"\n{'='*60}")
        print(f">> MSR: {cfg.DATASET.NAME}  classes={classnames}")
        print(f"{'='*60}")

        print(f"\n[1/4] Loading BiomedCLIP backbone...")
        biomedclip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        prec = cfg.TRAINER.MSR_BIOMEDCLIP.PREC
        if prec == "fp32" or prec == "amp":
            biomedclip_model = biomedclip_model.float()

        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        print(f"\n[2/4] Building MSR model...")
        self.model = MSR_BiomedCLIP_Model(
            cfg=cfg, classnames=classnames,
            biomedclip_model=biomedclip_model, tokenizer=tokenizer,
        )

        print(f"\n[3/4] Freezing BiomedCLIP backbone...")
        trainable_keywords = ['vis_prompts', 'text_ctx', 'W_coupling', 'agg_weights', 'log_contrastive_temp']
        for name, param in self.model.named_parameters():
            if not any(kw in name for kw in trainable_keywords):
                param.requires_grad_(False)

        self.model.to(self.device)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  [trainable] {name}: {param.numel():,}")

        print(f"\n[4/4] Building optimizer (LR={cfg.OPTIM.LR})...")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("msr_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"\nMultiple GPUs detected (n_gpus={device_count}), using DataParallel...")
            self.model = nn.DataParallel(self.model)

        print(f"\n{'='*60}")
        print(f"MSR build complete.")
        print(f"{'='*60}")

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MSR_BIOMEDCLIP.PREC

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

        loss_summary = {
            "loss": loss.item(), "loss_ce": outputs.get("loss_ce", 0),
            "loss_sr_align": outputs.get("loss_sr_align", 0),
            "loss_sr_contrast": outputs.get("loss_sr_contrast", 0), "acc": acc,
        }

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
