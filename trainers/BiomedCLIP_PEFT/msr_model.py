import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict


class MSR_BiomedCLIP_Model(nn.Module):

    def __init__(self, cfg, classnames, biomedclip_model, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.num_classes = len(classnames)
        self.tokenizer = tokenizer

        self.visual = biomedclip_model.visual
        self.dtype = biomedclip_model.text.transformer.dtype
        self.text_encoder = biomedclip_model.text
        self.logit_scale = biomedclip_model.logit_scale

        self.vit_dim = 768
        self.embed_dim = 512

        msr_cfg = cfg.TRAINER.MSR_BIOMEDCLIP
        self.sr_lambda = msr_cfg.SR_LAMBDA
        self.contrastive_lambda = msr_cfg.CONTRASTIVE_LAMBDA
        self.use_coupling = msr_cfg.USE_COUPLING
        self.use_aggregation = msr_cfg.USE_AGGREGATION
        self.use_contrastive = msr_cfg.USE_CONTRASTIVE

        # 1. Visual Prompts: Token-Level Coupling
        vpt_cfg = msr_cfg.VPT
        self.n_vis_prompts = vpt_cfg.N_PROMPTS
        self.vis_layer_idxs: List[int] = list(vpt_cfg.LAYER_IDXS)
        self.vis_layer_set = set(self.vis_layer_idxs)

        if self.use_coupling:
            self.W_coupling = nn.Parameter(
                torch.empty(self.n_vis_prompts, msr_cfg.TEXT_PROMPT.N_CTX)
            )
            nn.init.normal_(self.W_coupling, std=0.01)

        self.vis_prompts = nn.ParameterDict()
        for idx in self.vis_layer_idxs:
            delta = nn.Parameter(torch.empty(1, self.n_vis_prompts, self.vit_dim))
            nn.init.xavier_uniform_(delta)
            self.vis_prompts[f'layer_{idx}'] = delta

        # 2. Text Prompts
        txt_cfg = msr_cfg.TEXT_PROMPT
        self.n_ctx = txt_cfg.N_CTX
        ctx_dim = 768
        ctx_init = txt_cfg.CTX_INIT

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            prompt_ids = self.tokenizer(ctx_init)
            with torch.no_grad():
                word_embeds = self.text_encoder.transformer.get_input_embeddings()
                ctx_init_embeds = word_embeds(prompt_ids).type(self.dtype)
            ctx_vectors = ctx_init_embeds[0, 1:1 + self.n_ctx, :].clone()
        else:
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        self.text_ctx = nn.Parameter(ctx_vectors)
        self._x_token_id = None

        # 3. Multi-Level Feature Aggregation
        self.agg_layer_idxs: List[int] = list(msr_cfg.AGG_LAYER_IDXS)
        self.agg_layer_set = set(self.agg_layer_idxs)
        if self.use_aggregation and self.agg_layer_set:
            n_agg = len(self.agg_layer_idxs) + 1
            self.agg_weights = nn.Parameter(torch.zeros(n_agg))

        # 4. Contrastive Temperature
        init_temp = msr_cfg.get('CONTRASTIVE_TEMP', 0.07)
        self.log_contrastive_temp = nn.Parameter(torch.tensor(math.log(init_temp)))

        n_vis_params = sum(p.numel() for p in self.vis_prompts.parameters())
        n_txt_params = self.text_ctx.numel()
        n_couple_params = self.W_coupling.numel() if self.use_coupling else 0
        n_agg_params = self.agg_weights.numel() if hasattr(self, 'agg_weights') else 0
        n_total = n_vis_params + n_txt_params + n_couple_params + n_agg_params + 1
        print(f"[MSR] Total trainable: {n_total:,}")
        print(f"  Coupling: {self.use_coupling}  Agg: {self.use_aggregation}  Contrastive: {self.use_contrastive}")

    def _compute_coupled_vis_prompts(self) -> Dict[str, torch.Tensor]:
        """vis_base = W_coupling @ text_ctx + per-layer residuals."""
        if self.use_coupling:
            vis_base = self.W_coupling @ self.text_ctx
            vis_base = vis_base.unsqueeze(0)
            return {layer: vis_base + delta for layer, delta in self.vis_prompts.items()}
        else:
            return {layer: delta for layer, delta in self.vis_prompts.items()}

    def _forward_visual_vpt(self, image: torch.Tensor) -> torch.Tensor:
        """VPT-Deep with coupled prompts + multi-level CLS aggregation."""
        device = image.device
        dtype = self.dtype
        batch_size = image.shape[0]

        x = self.visual.trunk.patch_embed(image.type(dtype))
        cls_token = self.visual.trunk.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.trunk.pos_embed
        x = self.visual.trunk.pos_drop(x)
        x = self.visual.trunk.norm_pre(x)

        coupled_prompts = self._compute_coupled_vis_prompts()
        intermediate_cls: Dict[int, torch.Tensor] = {}

        for block_idx, block in enumerate(self.visual.trunk.blocks):
            if block_idx in self.vis_layer_set:
                prompts = coupled_prompts[f'layer_{block_idx}'].expand(batch_size, -1, -1)
                x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
                x = block(x)
                x = torch.cat([x[:, :1], x[:, 1 + self.n_vis_prompts:]], dim=1)
            else:
                x = block(x)

            if block_idx in self.agg_layer_set:
                intermediate_cls[block_idx] = x[:, 0]

        final_cls = x[:, 0]

        if self.use_aggregation and self.agg_layer_set:
            agg_features = [intermediate_cls[idx] for idx in self.agg_layer_idxs]
            agg_features.append(final_cls)
            features_stack = torch.stack(agg_features, dim=0)
            weights = F.softmax(self.agg_weights, dim=0)
            aggregated = (weights[:, None, None] * features_stack).sum(dim=0)
            x_norm = self.visual.trunk.norm(aggregated)
            adapted = self.visual.head(x_norm)
        else:
            x = self.visual.trunk.norm(x)
            adapted = self.visual.head(x[:, 0])

        return F.normalize(adapted, dim=-1)

    def _get_x_token_id(self):
        if self._x_token_id is None:
            x_tokens = self.tokenizer("X")
            self._x_token_id = x_tokens[0, 1].item()
        return self._x_token_id

    def _forward_text_prompt(self, classnames=None) -> torch.Tensor:
        """Replace X tokens with learnable text_ctx, forward through BERT."""
        if classnames is None:
            classnames = self.classnames

        device = self.text_ctx.device
        n_cls = len(classnames)

        prompt_prefix = " ".join(["X"] * self.n_ctx)
        prompts = [prompt_prefix + " " + name.replace("_", " ") + "." for name in classnames]
        token_ids = torch.cat([self.tokenizer(p) for p in prompts]).to(device)

        word_embeddings = self.text_encoder.transformer.get_input_embeddings()
        x_id = self._get_x_token_id()

        with torch.no_grad():
            full_embeds = word_embeddings(token_ids).float()

        for i in range(n_cls):
            x_positions = (token_ids[i] == x_id).nonzero(as_tuple=True)[0]
            if len(x_positions) >= self.n_ctx + 1:
                full_embeds[i, x_positions[1:1 + self.n_ctx]] = self.text_ctx.to(full_embeds.dtype)

        text_embeds = self.text_encoder(full_embeds, inputs_embeds=True, y=token_ids)
        return F.normalize(text_embeds, dim=-1)

    def _compute_loss_sr(self, v_feat, t_feat, label) -> Dict[str, torch.Tensor]:
        """MSE alignment (stop-grad on text) + optional InfoNCE contrastive."""
        v_norm = F.normalize(v_feat, dim=-1)
        t_pos = t_feat[label]
        t_norm = F.normalize(t_pos, dim=-1)

        loss_align = F.mse_loss(v_norm, t_norm.detach())

        if self.use_contrastive:
            sim = v_norm @ t_norm.t()
            sim = sim / torch.exp(self.log_contrastive_temp)
            batch_labels = torch.arange(sim.size(0), device=sim.device)
            loss_contrast = F.cross_entropy(sim, batch_labels)
        else:
            loss_contrast = v_norm.new_tensor(0.0)

        return {"loss_sr_align": loss_align, "loss_sr_contrast": loss_contrast}

    def forward(self, image: torch.Tensor, label=None) -> dict:
        visual_features = self._forward_visual_vpt(image)
        text_features = self._forward_text_prompt()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features @ text_features.t()

        output = {"logits": logits, "visual_features": visual_features, "text_features": text_features}

        if label is not None:
            loss_ce = F.cross_entropy(logits, label)
            loss_dict = self._compute_loss_sr(visual_features, text_features, label)
            loss = loss_ce + self.sr_lambda * loss_dict["loss_sr_align"] + self.contrastive_lambda * loss_dict["loss_sr_contrast"]
            output["loss"] = loss
            output["loss_ce"] = loss_ce.item()
            output["loss_sr_align"] = loss_dict["loss_sr_align"].item()
            output["loss_sr_contrast"] = loss_dict["loss_sr_contrast"].item()

        return output
