import torch
import torch.nn as nn
import torch.nn.functional as F


class MaPLe_BiomedCLIP_Model(nn.Module):
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

        maple_cfg = cfg.TRAINER.MAPLE_BIOMEDCLIP

        # 1. Visual Prompts: VPT-Deep with coupling
        vpt_cfg = maple_cfg.VPT
        self.n_vis_prompts = vpt_cfg.N_PROMPTS
        self.vis_layer_idxs = list(vpt_cfg.LAYER_IDXS)
        self.vis_layer_set = set(self.vis_layer_idxs)

        # Coupling weight: vis_base = W_coupling @ text_ctx
        self.W_coupling = nn.Parameter(
            torch.empty(self.n_vis_prompts, maple_cfg.TEXT_PROMPT.N_CTX)
        )
        nn.init.normal_(self.W_coupling, std=0.01)

        # Per-layer residuals
        self.vis_prompts = nn.ParameterDict()
        for idx in self.vis_layer_idxs:
            delta = nn.Parameter(torch.empty(1, self.n_vis_prompts, self.vit_dim))
            nn.init.xavier_uniform_(delta)
            self.vis_prompts[f'layer_{idx}'] = delta

        # 2. Text Prompts
        txt_cfg = maple_cfg.TEXT_PROMPT
        self.n_ctx = txt_cfg.N_CTX
        ctx_dim = 768

        ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.text_ctx = nn.Parameter(ctx_vectors)
        self._x_token_id = None

        n_vis = sum(p.numel() for p in self.vis_prompts.parameters())
        n_txt = self.text_ctx.numel()
        n_couple = self.W_coupling.numel()
        n_total = n_vis + n_txt + n_couple
        print(f"[MaPLe] VPT: {n_vis:,} | Text ctx: {n_txt:,} | Coupling: {n_couple:,}")
        print(f"[MaPLe] Total trainable: {n_total:,}  layers: {self.vis_layer_idxs}")

    def _compute_coupled_vis_prompts(self):
        """vis_base = W_coupling @ text_ctx + per-layer residual."""
        vis_base = self.W_coupling @ self.text_ctx
        vis_base = vis_base.unsqueeze(0)
        return {
            layer: vis_base + delta
            for layer, delta in self.vis_prompts.items()
        }

    def _forward_visual_vpt(self, image):
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

        for block_idx, block in enumerate(self.visual.trunk.blocks):
            if block_idx in self.vis_layer_set:
                prompts = coupled_prompts[f'layer_{block_idx}'].expand(batch_size, -1, -1)
                x = torch.cat([x[:, :1], prompts, x[:, 1:]], dim=1)
                x = block(x)
                x = torch.cat([x[:, :1], x[:, 1 + self.n_vis_prompts:]], dim=1)
            else:
                x = block(x)

        x = self.visual.trunk.norm(x)
        adapted = self.visual.head(x[:, 0])
        return F.normalize(adapted, dim=-1)

    def _get_x_token_id(self):
        if self._x_token_id is None:
            x_tokens = self.tokenizer("X")
            self._x_token_id = x_tokens[0, 1].item()
        return self._x_token_id

    def _forward_text_prompt(self, classnames=None):
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

    def forward(self, image, label=None):
        visual_features = self._forward_visual_vpt(image)
        text_features = self._forward_text_prompt()

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * visual_features @ text_features.t()

        output = {"logits": logits, "visual_features": visual_features}
        if label is not None:
            output["loss"] = F.cross_entropy(logits, label)
        return output
