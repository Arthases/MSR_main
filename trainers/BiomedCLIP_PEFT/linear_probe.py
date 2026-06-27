import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.metrics import compute_accuracy

from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer


class LinearProbeHead(nn.Module):
    """Linear layer mapping visual features to class logits."""

    def __init__(self, embed_dim: int, num_classes: int, bias_zero: bool = True):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        if bias_zero:
            nn.init.zeros_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class BiomedCLIP_Model(nn.Module):
    """Frozen BiomedCLIP visual encoder + trainable linear probe head."""

    def __init__(self, cfg, classnames, biomedclip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)

        self.visual = biomedclip_model.visual
        self.dtype = biomedclip_model.text.transformer.dtype

        dummy = torch.zeros(1, 3, 224, 224).to(next(self.visual.parameters()).device)
        with torch.no_grad():
            out = self.visual(dummy.type(self.dtype))
        self.embed_dim = out.shape[1]

        self.head = LinearProbeHead(self.embed_dim, self.num_classes, bias_zero=True)

        print(f"[BiomedCLIP_Model] Visual embed_dim: {self.embed_dim}")
        print(f"[BiomedCLIP_Model] Num classes: {self.num_classes}")

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            visual_features = self.visual(image.type(self.dtype))
            visual_features = visual_features / visual_features.norm(dim=-1, keepdim=True)
        return visual_features

    def forward(self, image: torch.Tensor, label=None) -> dict:
        visual_features = self.encode_image(image)
        logits = self.head(visual_features)
        output = {"logits": logits, "visual_features": visual_features}
        if label is not None:
            output["loss"] = F.cross_entropy(logits, label)
        return output


@TRAINER_REGISTRY.register()
class BiomedCLIP(TrainerX):
    """Linear Probe trainer: freeze backbone, train linear head."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.BIOMEDCLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"\n>> LP: {cfg.DATASET.NAME}  classes={classnames}")
        print(f"\nLoading BiomedCLIP...")
        biomedclip_model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        if cfg.TRAINER.BIOMEDCLIP.PREC == "fp32" or cfg.TRAINER.BIOMEDCLIP.PREC == "amp":
            biomedclip_model = biomedclip_model.float()

        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            cache_dir='clip/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        print("\nBuilding Linear Probe model...")
        self.model = BiomedCLIP_Model(
            cfg=cfg, classnames=classnames,
            biomedclip_model=biomedclip_model, tokenizer=tokenizer,
        )

        print("\nFreezing BiomedCLIP (visual and text encoders)...")
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad_(False)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.4f}%)")

        self.model.to(self.device)

        print(f"\nBuilding optimizer (LR={cfg.OPTIM.LR})...")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("lp_head", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.BIOMEDCLIP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"\nMultiple GPUs detected (n_gpus={device_count}), using DataParallel...")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.BIOMEDCLIP.PREC

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
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def model_inference(self, input):
        return self.model(input, label=None)["logits"]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = f"{directory}/{name}/{model_file}"
            if not __import__('os').path.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                new_state_dict[k] = v
            state_dict = new_state_dict

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
