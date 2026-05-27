# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging

import torch
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from torch import Tensor, nn
from torch.nn import functional as F

from .position_encoding_3d import (
    PositionEmbeddingSine3D,  # Original 2D, we will define 3D below.
)

TRANSFORMER_DECODER_REGISTRY = Registry('TRANSFORMER_MODULE')
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in MaskFormer.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(
        cfg, in_channels, mask_classification
    )


class SelfAttentionLayer(nn.Module):
    # ... (unchanged, as it operates on sequences)
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation='relu',
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    # ... (unchanged, as it operates on sequences)
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation='relu',
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        pos: Tensor | None = None,
        query_pos: Tensor | None = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    # ... (unchanged, as it operates on sequences)
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation='relu',
        normalize_before=False,
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Tensor | None):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):
    _version = 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if 'static_query' in k:
                    newk = k.replace('static_query', 'query_feat')
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f'Weight format of {self.__class__.__name__} have changed! '
                    'Please upgrade your models. Applying automatic conversion now ...'
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        dn: str = 'no',
        noise_scale: float = 0.4,
        dn_num: int = 100,
        # ------------------------------------------------------------
        # New: MaskDINO-style feature-map query initialization
        # ------------------------------------------------------------
        feature_query_init: bool = True,
        feature_query_init_detach: bool = True,
        feature_query_init_add_learned_content: bool = False,
        feature_query_init_add_learned_pos: bool = False,
    ):
        """
        Args:
            in_channels: channels of the input features.
            mask_classification: whether to add mask classifier or not.
            num_classes: number of classes.
            hidden_dim: Transformer feature dimension.
            num_queries: number of queries.
            nheads: number of heads.
            dim_feedforward: feature dimension in FFN.
            dec_layers: number of Transformer decoder layers.
            pre_norm: whether to use pre-LayerNorm or not.
            mask_dim: mask feature dimension.
            enforce_input_project: add input projection even if channels match.

            feature_query_init:
                If True, initialize normal matching queries from multi-scale feature map
                top-K tokens, MaskDINO-style.

            feature_query_init_detach:
                If True, detach selected feature tokens before feeding them as query content.
                MaskDINO detaches selected encoder features for decoder queries after
                producing intermediate predictions. In this simplified Mask2Former version,
                default False is more practical because we do not add an intermediate
                encoder proposal loss here.

            feature_query_init_add_learned_content:
                If True, add original learned query_feat as a residual to selected feature tokens.

            feature_query_init_add_learned_pos:
                If True, add original learned query_embed as a residual to coordinate-based query position.
        """
        super().__init__()

        assert mask_classification, 'Only support mask classification model'
        self.mask_classification = mask_classification

        # 3D positional encoding
        assert hidden_dim % 6 == 0, (
            'hidden_dim must be divisible by 6 for 3D positional encoding'
        )
        self.pe_layer = PositionEmbeddingSine3D(
            d_model=hidden_dim, normalize=True
        )

        # Transformer decoder layers
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_classes = num_classes

        self.dn = dn
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.hidden_dim = hidden_dim

        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.dn_pos_embed = nn.Sequential(
            MLP(6, hidden_dim, hidden_dim, 3),
            nn.LayerNorm(hidden_dim),
        )

        self.debug_vis_enabled = False
        self.debug_vis_topk = 300

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries

        # Original learned queries are kept:
        # 1) as fallback when feature_query_init=False
        # 2) optionally as residual identity embeddings
        # 3) for checkpoint compatibility
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        # input projection
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
                nn.init.kaiming_uniform_(proj.weight, a=1)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
                self.input_proj.append(proj)
            else:
                self.input_proj.append(nn.Sequential())

        # ------------------------------------------------------------
        # New: modules for feature-map query initialization
        # ------------------------------------------------------------
        self.feature_query_init = bool(feature_query_init)
        self.feature_query_init_detach = bool(feature_query_init_detach)
        self.feature_query_init_add_learned_content = bool(
            feature_query_init_add_learned_content
        )
        self.feature_query_init_add_learned_pos = bool(
            feature_query_init_add_learned_pos
        )

        # MaskDINO-style encoder output transform before selecting top-K tokens.
        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        # Convert selected token coordinate box:
        # [center_z, center_y, center_x, size_z, size_y, size_x]
        # into hidden_dim query position.
        self.feature_query_pos_embed = nn.Sequential(
            MLP(6, hidden_dim, hidden_dim, 3),
            nn.LayerNorm(hidden_dim),
        )

        # output heads
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret['in_channels'] = in_channels
        ret['mask_classification'] = mask_classification

        ret['num_classes'] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret['hidden_dim'] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret['num_queries'] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret['nheads'] = cfg.MODEL.MASK_FORMER.NHEADS
        ret['dim_feedforward'] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret['dec_layers'] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret['pre_norm'] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret['enforce_input_project'] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret['mask_dim'] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret['dn'] = cfg.MODEL.MASK_FORMER.DN
        ret['noise_scale'] = cfg.MODEL.MASK_FORMER.DN_NOISE_SCALE
        ret['dn_num'] = cfg.MODEL.MASK_FORMER.DN_NUM

        # ------------------------------------------------------------
        # New config options.
        # If these keys are not in your yaml/config, the defaults below are used.
        # ------------------------------------------------------------
        ret['feature_query_init'] = getattr(
            cfg.MODEL.MASK_FORMER,
            'FEATURE_QUERY_INIT',
            True,
        )
        ret['feature_query_init_detach'] = getattr(
            cfg.MODEL.MASK_FORMER,
            'FEATURE_QUERY_INIT_DETACH',
            True,
        )
        ret['feature_query_init_add_learned_content'] = getattr(
            cfg.MODEL.MASK_FORMER,
            'FEATURE_QUERY_INIT_ADD_LEARNED_CONTENT',
            False,
        )
        ret['feature_query_init_add_learned_pos'] = getattr(
            cfg.MODEL.MASK_FORMER,
            'FEATURE_QUERY_INIT_ADD_LEARNED_POS',
            False,
        )

        return ret

    def _make_normalized_coord_boxes_3d(self, d, h, w, device, dtype):
        """
        Build one normalized coordinate-box for every token in a 3D feature map.

        Return:
            coord_boxes: [d*h*w, 6]
                [center_z, center_y, center_x, size_z, size_y, size_x]
                all values in [0, 1].

        This is not a real object bbox.
        It is a token-level anchor box indicating where this feature token is located.
        """
        z = (torch.arange(d, device=device, dtype=dtype) + 0.5) / max(d, 1)
        y = (torch.arange(h, device=device, dtype=dtype) + 0.5) / max(h, 1)
        x = (torch.arange(w, device=device, dtype=dtype) + 0.5) / max(w, 1)

        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

        sz = torch.full_like(zz, 1.0 / max(d, 1))
        sy = torch.full_like(yy, 1.0 / max(h, 1))
        sx = torch.full_like(xx, 1.0 / max(w, 1))

        coord_boxes = torch.stack([zz, yy, xx, sz, sy, sx], dim=-1)
        coord_boxes = coord_boxes.reshape(-1, 6)
        coord_boxes = coord_boxes.clamp(0.0, 1.0)

        return coord_boxes

    def _gather_tokens_by_index(self, tokens, indices):
        """
        Args:
            tokens:  [B, S, C]
            indices: [B, Q]

        Return:
            selected: [B, Q, C]
        """
        return torch.gather(
            tokens,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )

    def masks_to_boxes_3d(self, masks, threshold=0.5):
        # masks: [B, Q, D, H, W], probability masks
        B, Q, D, H, W = masks.shape
        boxes = masks.new_zeros(B, Q, 6)

        for b in range(B):
            for q in range(Q):
                m = masks[b, q] > threshold
                if m.sum() == 0:
                    boxes[b, q] = masks.new_tensor(
                        [0.5, 0.5, 0.5, 1.0 / D, 1.0 / H, 1.0 / W]
                    )
                    continue

                coords = torch.nonzero(
                    m, as_tuple=False
                ).float()  # [N, 3], zyx
                zyx_min = coords.min(dim=0).values
                zyx_max = coords.max(dim=0).values

                center = (zyx_min + zyx_max) * 0.5
                size = zyx_max - zyx_min + 1.0

                norm = masks.new_tensor(
                    [
                        max(D - 1, 1),
                        max(H - 1, 1),
                        max(W - 1, 1),
                        max(D, 1),
                        max(H, 1),
                        max(W, 1),
                    ]
                )

                box = torch.cat([center, size], dim=0) / norm
                boxes[b, q] = box.clamp(0.0, 1.0)

        return boxes

    def _build_feature_initialized_queries(
        self,
        src,
        pos,
        coord_boxes,
        bs,
        mask_features,
    ):
        """
        MaskDINO-style top-K encoder proposal query initialization.

        Main logic:
            multi-scale tokens
            -> encoder output transform
            -> dense class logits
            -> top-K proposals
            -> selected proposal masks
            -> mask2box
            -> query content / query position
        """
        # ------------------------------------------------------------
        # 1. Flatten all feature levels into one dense token sequence.
        # ------------------------------------------------------------
        memory = torch.cat(
            [s.permute(1, 0, 2) for s in src],
            dim=1,
        )  # [B, S_total, C]

        # ------------------------------------------------------------
        # 2. Encoder output transform.
        # ------------------------------------------------------------
        memory_for_select = self.enc_output_norm(
            self.enc_output(memory)
        )  # [B, S_total, C]

        # ------------------------------------------------------------
        # 3. Dense classification logits for top-K proposal selection.
        # ------------------------------------------------------------
        enc_logits = self.class_embed(
            memory_for_select
        )  # [B, S_total, num_classes + 1]

        if enc_logits.shape[-1] == self.num_classes + 1:
            dense_scores = (
                enc_logits[..., : self.num_classes].max(dim=-1).values
            )
        else:
            dense_scores = enc_logits.max(dim=-1).values

        topk = min(self.num_queries, memory_for_select.shape[1])

        topk_indices = torch.topk(
            dense_scores,
            k=topk,
            dim=1,
            sorted=False,
        ).indices  # [B, topk]

        # ------------------------------------------------------------
        # 4. Gather top-K encoder features as decoder query content.
        # ------------------------------------------------------------
        selected_memory = self._gather_tokens_by_index(
            memory_for_select,
            topk_indices,
        )  # [B, topk, C]

        if self.feature_query_init_detach:
            selected_memory = selected_memory.detach()

        # ------------------------------------------------------------
        # 5. Gather top-K encoder logits.
        #    These are used as enc_outputs["pred_logits"].
        # ------------------------------------------------------------
        selected_enc_logits = self._gather_tokens_by_index(
            enc_logits,
            topk_indices,
        )  # [B, topk, num_classes + 1]

        # ------------------------------------------------------------
        # 6. Only generate masks for top-K selected proposals.
        #    Do NOT generate [B, S_total, D, H, W] masks.
        # ------------------------------------------------------------
        enc_mask_embed = self.mask_embed(
            memory_for_select
        )  # [B, S_total, mask_dim]

        selected_mask_embed = self._gather_tokens_by_index(
            enc_mask_embed,
            topk_indices,
        )  # [B, topk, mask_dim]

        selected_masks = torch.einsum(
            'bqc,bcdhw->bqdhw',
            selected_mask_embed,
            mask_features,
        )  # [B, topk, D, H, W]

        # ------------------------------------------------------------
        # 7. MaskDINO-style mask2box initialization for query position.
        # ------------------------------------------------------------
        selected_boxes = self.masks_to_boxes_3d(
            selected_masks.sigmoid()
        )  # [B, topk, 6]

        selected_boxes = selected_boxes.detach()

        selected_query_pos = self.feature_query_pos_embed(
            selected_boxes
        )  # [B, topk, C]

        # ------------------------------------------------------------
        # 8. Optional learned residuals.
        #    Keep False if you want to stay close to MaskDINO.
        # ------------------------------------------------------------
        if self.feature_query_init_add_learned_content:
            selected_memory = selected_memory + self.query_feat.weight[
                :topk
            ].unsqueeze(0).to(selected_memory.dtype)

        if self.feature_query_init_add_learned_pos:
            selected_query_pos = selected_query_pos + self.query_embed.weight[
                :topk
            ].unsqueeze(0).to(selected_query_pos.dtype)

        # ------------------------------------------------------------
        # 9. Fallback if total dense tokens < num_queries.
        #    Usually this will not happen.
        # ------------------------------------------------------------
        if topk < self.num_queries:
            remain = self.num_queries - topk

            learned_content = self.query_feat.weight[topk : topk + remain]
            learned_content = learned_content.unsqueeze(0).expand(bs, -1, -1)
            learned_content = learned_content.to(
                device=selected_memory.device,
                dtype=selected_memory.dtype,
            )

            learned_pos = self.query_embed.weight[topk : topk + remain]
            learned_pos = learned_pos.unsqueeze(0).expand(bs, -1, -1)
            learned_pos = learned_pos.to(
                device=selected_query_pos.device,
                dtype=selected_query_pos.dtype,
            )

            selected_memory = torch.cat(
                [selected_memory, learned_content], dim=1
            )
            selected_query_pos = torch.cat(
                [selected_query_pos, learned_pos], dim=1
            )

            # pad encoder outputs only for shape consistency
            pad_logits = selected_enc_logits.new_zeros(
                bs,
                remain,
                selected_enc_logits.shape[-1],
            )
            pad_masks = selected_masks.new_zeros(
                bs,
                remain,
                selected_masks.shape[-3],
                selected_masks.shape[-2],
                selected_masks.shape[-1],
            )

            selected_enc_logits = torch.cat(
                [selected_enc_logits, pad_logits], dim=1
            )
            selected_masks = torch.cat([selected_masks, pad_masks], dim=1)

        # [B, Q, C] -> [Q, B, C]
        output = selected_memory.transpose(0, 1).contiguous()
        query_embed = selected_query_pos.transpose(0, 1).contiguous()

        enc_outputs = {
            'pred_logits': selected_enc_logits,
            'pred_masks': selected_masks,
        }

        return output, query_embed, enc_outputs

    def prepare_for_dn(self, targets, batch_size, device):
        if (not self.training) or self.dn == 'no':
            return None, None, None, None

        scalar = self.dn_num
        noise_scale = self.noise_scale

        known_num = [len(t['labels']) for t in targets]
        max_known = max(known_num) if len(known_num) > 0 else 0

        if max_known == 0:
            return None, None, None, None

        scalar = scalar // max_known
        if scalar == 0:
            return None, None, None, None

        boxes_all = []
        labels_all = []
        batch_idx_all = []

        for b, t in enumerate(targets):
            masks = t['masks'].to(device)  # [num_inst, D, H, W]
            labels = t['labels'].long().to(device)

            if masks.numel() == 0:
                continue

            num_inst = masks.shape[0]
            D, H, W = masks.shape[-3:]
            norm = torch.tensor(
                [
                    max(D - 1, 1),
                    max(H - 1, 1),
                    max(W - 1, 1),
                    max(D - 1, 1),
                    max(H - 1, 1),
                    max(W - 1, 1),
                ],
                device=device,
                dtype=torch.float32,
            )

            boxes = []
            for i in range(num_inst):
                m = masks[i] > 0

                if m.sum() == 0:
                    # fallback: center in middle, very small box
                    boxes.append(
                        torch.tensor(
                            [
                                0.5,
                                0.5,
                                0.5,
                                1.0 / max(D, 1),
                                1.0 / max(H, 1),
                                1.0 / max(W, 1),
                            ],
                            device=device,
                            dtype=torch.float32,
                        )
                    )
                    continue

                coords = torch.nonzero(
                    m, as_tuple=False
                ).float()  # [K, 3], zyx

                zyx_min = coords.min(dim=0).values
                zyx_max = coords.max(dim=0).values

                center_zyx = (zyx_min + zyx_max) * 0.5
                size_zyx = zyx_max - zyx_min + 1.0

                box_zyx = torch.cat([center_zyx, size_zyx], dim=0) / norm
                box_zyx = box_zyx.clamp(0.0, 1.0)

                boxes.append(box_zyx)

            boxes = torch.stack(boxes, dim=0)  # [num_inst, 6]

            boxes_all.append(boxes)
            labels_all.append(labels)
            batch_idx_all.append(
                torch.full((num_inst,), b, device=device, dtype=torch.long)
            )

        if len(boxes_all) == 0:
            return None, None, None, None

        boxes = torch.cat(boxes_all, dim=0)  # [N_gt, 6]
        labels = torch.cat(labels_all, dim=0).long()  # [N_gt]
        batch_idx = torch.cat(batch_idx_all, dim=0)  # [N_gt]

        known_boxes = boxes.repeat(scalar, 1)
        known_labels = labels.repeat(scalar)
        known_bid = batch_idx.repeat(scalar)

        if noise_scale > 0:
            known_boxes_expand = known_boxes.clone()

            diff = torch.zeros_like(known_boxes_expand)
            diff[:, :3] = known_boxes_expand[:, 3:] * 0.5
            diff[:, 3:] = known_boxes_expand[:, 3:]

            noise = (
                (torch.rand_like(known_boxes_expand) * 2.0 - 1.0)
                * diff
                * noise_scale
            )
            known_boxes_expand = (known_boxes_expand + noise).clamp(0.0, 1.0)

            # 防止尺寸被扰动到 0
            known_boxes_expand[:, 3:] = known_boxes_expand[:, 3:].clamp(
                min=1e-4, max=1.0
            )
        else:
            known_boxes_expand = known_boxes

        known_labels_expand = known_labels.clamp(
            min=0, max=self.num_classes - 1
        )

        input_label_embed = self.label_enc(
            known_labels_expand
        )  # content query
        input_pos_embed = self.dn_pos_embed(
            known_boxes_expand
        )  # bbox position query
        input_pos_embed = input_pos_embed.to(input_label_embed.dtype)
        dtype = input_label_embed.dtype
        single_pad = int(max_known)
        pad_size = int(single_pad * scalar)

        # # ---------- DEBUG: save DN noisy bbox overlay on raw image ----------
        # if not hasattr(self, "_dn_bbox_overlay_saved"):
        #     self._dn_bbox_overlay_saved = True

        #     import os
        #     import tifffile
        #     import torch.nn.functional as F

        #     save_path = "./debug_dn_bbox_overlay.tif"
        #     alpha = 0.35

        #     b0 = 0
        #     D, H, W = targets[b0]["masks"].shape[-3:]

        #     # 用 GT masks 合成一个灰度背景；如果你想叠到原始图像上，更推荐在主函数里做
        #     bg = targets[b0]["masks"].any(dim=0).float()  # [D,H,W]
        #     bg = (bg * 255).byte()

        #     rgb = torch.stack([bg, bg, bg], dim=-1).float() / 255.0  # [D,H,W,3]

        #     debug_boxes = known_boxes_expand[known_bid == b0].detach().float().cpu()

        #     for box in debug_boxes:
        #         cz, cy, cx, dz, dy, dx = box.tolist()

        #         z0 = int((cz - dz / 2) * D)
        #         z1 = int((cz + dz / 2) * D)
        #         y0 = int((cy - dy / 2) * H)
        #         y1 = int((cy + dy / 2) * H)
        #         x0 = int((cx - dx / 2) * W)
        #         x1 = int((cx + dx / 2) * W)

        #         z0, z1 = max(z0, 0), min(z1, D - 1)
        #         y0, y1 = max(y0, 0), min(y1, H - 1)
        #         x0, x1 = max(x0, 0), min(x1, W - 1)

        #         if z1 <= z0 or y1 <= y0 or x1 <= x0:
        #             continue

        #         # 只画 bbox 边框，不填充
        #         edge = torch.zeros((D, H, W), dtype=torch.bool)

        #         edge[z0, y0:y1+1, x0:x1+1] = True
        #         edge[z1, y0:y1+1, x0:x1+1] = True
        #         edge[z0:z1+1, y0, x0:x1+1] = True
        #         edge[z0:z1+1, y1, x0:x1+1] = True
        #         edge[z0:z1+1, y0:y1+1, x0] = True
        #         edge[z0:z1+1, y0:y1+1, x1] = True

        #         # 红色半透明框
        #         rgb[edge] = rgb[edge] * (1 - alpha) + torch.tensor(
        #             [1.0, 0.0, 0.0],
        #             device=rgb.device,
        #             dtype=rgb.dtype,
        #         ) * alpha

        #     overlay = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()

        #     tifffile.imwrite(
        #         save_path,
        #         overlay,              # [D,H,W,3]
        #         imagej=True,
        #     )

        #     print(f"[DN DEBUG] saved bbox overlay to {save_path}")

        padding_label = torch.zeros(
            pad_size, self.hidden_dim, device=device, dtype=dtype
        )
        padding_pos = torch.zeros(
            pad_size, self.hidden_dim, device=device, dtype=dtype
        )

        input_query_label = padding_label.unsqueeze(0).repeat(batch_size, 1, 1)
        input_query_embed = padding_pos.unsqueeze(0).repeat(batch_size, 1, 1)

        map_known_indice = (
            torch.cat([torch.arange(num, device=device) for num in known_num])
            if sum(known_num) > 0
            else torch.tensor([], device=device, dtype=torch.long)
        )

        map_known_indice = torch.cat(
            [map_known_indice + single_pad * i for i in range(scalar)]
        ).long()

        input_query_label[(known_bid.long(), map_known_indice)] = (
            input_label_embed
        )
        input_query_embed[(known_bid.long(), map_known_indice)] = (
            input_pos_embed
        )

        tgt_size = pad_size + self.num_queries
        self_attn_mask = torch.zeros(tgt_size, tgt_size, device=device).bool()

        self_attn_mask[pad_size:, :pad_size] = True

        for i in range(scalar):
            start = single_pad * i
            end = single_pad * (i + 1)
            self_attn_mask[start:end, :start] = True
            self_attn_mask[start:end, end:pad_size] = True

        mask_dict = {
            'known_indice': torch.arange(
                len(known_labels), device=device
            ).long(),
            'batch_idx': batch_idx.long(),
            'known_bid': known_bid.long(),
            'map_known_indice': map_known_indice.long(),
            'known_labels': known_labels.long(),
            'known_boxes': known_boxes,
            'known_boxes_noisy': known_boxes_expand,
            'known_lbs_bboxes': (known_labels.long(), known_boxes),
            'pad_size': pad_size,
            'scalar': scalar,
            'known_num': known_num,
        }

        return input_query_label, input_query_embed, self_attn_mask, mask_dict

    def dn_post_process(self, predictions_class, predictions_mask, mask_dict):
        pad_size = mask_dict['pad_size']

        predictions_class = torch.stack(
            predictions_class
        )  # [L, B, Q_all, C+1]
        predictions_mask = torch.stack(
            predictions_mask
        )  # [L, B, Q_all, D, H, W]

        output_known_class = predictions_class[:, :, :pad_size, :]
        output_known_mask = predictions_mask[:, :, :pad_size]

        predictions_class = predictions_class[:, :, pad_size:, :]
        predictions_mask = predictions_mask[:, :, pad_size:]

        out = {
            'pred_logits': output_known_class[-1],
            'pred_masks': output_known_mask[-1],
            'aux_outputs': self._set_aux_loss(
                list(output_known_class[:-1]),
                list(output_known_mask[:-1]),
            ),
        }

        mask_dict['output_known_lbs_bboxes'] = out

        return list(predictions_class), list(predictions_mask)

    def forward(self, x, mask_features, mask=None, targets=None):
        """
        Args:
            x:
                list of multi-scale 3D feature maps.
                Each element shape: [B, C, D, H, W]

            mask_features:
                [B, mask_dim, Dm, Hm, Wm]

            targets:
                used for denoising training.

        Returns:
            out:
                {
                    "pred_logits",
                    "pred_masks",
                    "aux_outputs",
                    optional debug fields
                }
            mask_dict:
                DN metadata if denoising is enabled.
        """
        assert len(x) == self.num_feature_levels

        src = []
        pos = []
        size_list = []
        coord_boxes = []

        del mask

        # ------------------------------------------------------------
        # 1. Build multi-scale projected features and positional encodings
        # ------------------------------------------------------------
        for i in range(self.num_feature_levels):
            d_i, h_i, w_i = x[i].shape[-3:]
            size_list.append((d_i, h_i, w_i))

            # 3D sine positional encoding: [B, C, D, H, W]
            pe = self.pe_layer(x[i])
            pe = pe.flatten(2)  # [B, C, S]

            # Project feature to hidden_dim
            projected = self.input_proj[i](x[i])  # [B, C, D, H, W]
            projected = projected.flatten(2)  # [B, C, S]

            # Add level embedding
            level_emb = self.level_embed.weight[i].view(1, -1, 1)
            projected = projected + level_emb

            # Build normalized coordinate boxes for this feature level
            coord_i = self._make_normalized_coord_boxes_3d(
                d=d_i,
                h=h_i,
                w=w_i,
                device=projected.device,
                dtype=projected.dtype,
            )  # [S, 6]
            coord_boxes.append(coord_i)

            # Convert to sequence format: [S, B, C]
            pos.append(pe.permute(2, 0, 1).contiguous())
            src.append(projected.permute(2, 0, 1).contiguous())

        _, bs, _ = src[0].shape
        enc_outputs = None
        # ------------------------------------------------------------
        # 2. Initialize normal matching queries
        # ------------------------------------------------------------
        if self.feature_query_init:
            # New: MaskDINO-style top-K feature initialization
            output, query_embed, enc_outputs = (
                self._build_feature_initialized_queries(
                    src=src,
                    pos=pos,
                    coord_boxes=coord_boxes,
                    bs=bs,
                    mask_features=mask_features,
                )
            )
        else:
            # Original Mask2Former learned queries
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        mask_dict = None
        dn_self_attn_mask = None

        # ------------------------------------------------------------
        # 3. Prepare DN queries and prepend them
        # ------------------------------------------------------------
        if self.training and self.dn != 'no':
            assert targets is not None
            (
                input_query_label,
                input_query_embed,
                dn_self_attn_mask,
                mask_dict,
            ) = self.prepare_for_dn(
                targets,
                bs,
                src[0].device,
            )

            if mask_dict is not None:
                # [B, pad, C] -> [pad, B, C]
                input_query_label = input_query_label.transpose(
                    0, 1
                ).contiguous()
                input_query_embed = input_query_embed.transpose(
                    0, 1
                ).contiguous()

                # Same as your current DN design:
                # content = label embedding + small positional hint
                dn_content = input_query_label + 0.1 * input_query_embed

                output = torch.cat([dn_content, output], dim=0)
                query_embed = torch.cat(
                    [input_query_embed, query_embed], dim=0
                )

        predictions_class = []
        predictions_mask = []

        debug_topk_masks = []
        debug_topk_scores = []
        debug_topk_indices = []

        # ------------------------------------------------------------
        # 4. Initial prediction before decoder layers
        # ------------------------------------------------------------
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0],
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        if self.debug_vis_enabled:
            layer_masks, layer_scores, layer_indices = (
                self._collect_debug_topk(
                    outputs_class,
                    outputs_mask,
                )
            )
            debug_topk_masks.append(layer_masks)
            debug_topk_scores.append(layer_scores)
            debug_topk_indices.append(layer_indices)

        # ------------------------------------------------------------
        # 5. Decoder layers
        # ------------------------------------------------------------
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            # Avoid fully-masked rows in attention mask
            attn_mask[
                torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])
            ] = False

            # Cross-attention
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index],
                query_pos=query_embed,
            )

            # Self-attention
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=dn_self_attn_mask,
                tgt_key_padding_mask=None,
                query_pos=query_embed,
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # Prediction heads after this layer
            next_level = (i + 1) % self.num_feature_levels
            outputs_class, outputs_mask, attn_mask = (
                self.forward_prediction_heads(
                    output,
                    mask_features,
                    attn_mask_target_size=size_list[next_level],
                )
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

            if self.debug_vis_enabled:
                layer_masks, layer_scores, layer_indices = (
                    self._collect_debug_topk(
                        outputs_class,
                        outputs_mask,
                    )
                )
                debug_topk_masks.append(layer_masks)
                debug_topk_scores.append(layer_scores)
                debug_topk_indices.append(layer_indices)

        assert len(predictions_class) == self.num_layers + 1

        # ------------------------------------------------------------
        # 6. Split DN outputs from normal matching outputs
        # ------------------------------------------------------------
        if mask_dict is not None:
            predictions_class, predictions_mask = self.dn_post_process(
                predictions_class,
                predictions_mask,
                mask_dict,
            )
        elif self.training:
            # Keep label_enc participating when DN is not active.
            predictions_class[-1] = (
                predictions_class[-1] + 0.0 * self.label_enc.weight.sum()
            )

        # ------------------------------------------------------------
        # 7. Final output
        # ------------------------------------------------------------
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_mask,
            ),
        }

        if enc_outputs is not None:
            out['enc_outputs'] = enc_outputs

        if self.debug_vis_enabled:
            out['debug_topk_masks'] = debug_topk_masks
            out['debug_topk_scores'] = debug_topk_scores
            out['debug_topk_indices'] = debug_topk_indices

        return out, mask_dict

    def _collect_debug_topk(self, outputs_class, outputs_mask):
        """
        Args:
            outputs_class: [N, Q, C+1]
            outputs_mask:  [N, Q, D, H, W]   raw logits

        Returns:
            topk_masks_cpu:   list of len N, each [K, D, H, W] float32 (prob)
            topk_scores_cpu:  list of len N, each [K]
            topk_indices_cpu: list of len N, each [K]   query indices
        """
        # [N, Q, C]
        scores = F.softmax(outputs_class, dim=-1)[..., :-1]

        N, Q, C = scores.shape
        topk_masks_cpu = []
        topk_scores_cpu = []
        topk_indices_cpu = []

        for b in range(N):
            scores_b = scores[b]  # [Q, C]
            masks_b = outputs_mask[b]  # [Q, D, H, W]

            flat_scores = scores_b.flatten(0, 1)  # [Q*C]
            k = min(self.debug_vis_topk, flat_scores.numel())

            topk_scores, topk_flat_indices = flat_scores.topk(k, sorted=False)
            query_indices = topk_flat_indices // C  # [K]

            # 取 query 对应 mask；保存概率图而不是 raw logits
            topk_masks = masks_b[query_indices].sigmoid()

            topk_masks_cpu.append(topk_masks.detach().cpu())
            topk_scores_cpu.append(topk_scores.detach().cpu())
            topk_indices_cpu.append(query_indices.detach().cpu())

        return topk_masks_cpu, topk_scores_cpu, topk_indices_cpu

    def forward_prediction_heads(
        self, output, mask_features, attn_mask_target_size
    ):
        """
        Args:
            output: (Q, N, hidden_dim) tensor
            mask_features: (N, mask_dim, D, H, W) tensor
            attn_mask_target_size: tuple (D, H, W) target size for the attention mask
        Returns:
            outputs_class: (N, Q, num_classes+1)
            outputs_mask: (N, Q, D, H, W) mask predictions
            attn_mask: (N*num_heads, Q, D*H*W) boolean mask for cross-attention
        """
        decoder_output = self.decoder_norm(output)  # (Q, N, hidden_dim)
        decoder_output = decoder_output.transpose(0, 1)  # (N, Q, hidden_dim)
        outputs_class = self.class_embed(
            decoder_output
        )  # (N, Q, num_classes+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum(
            'nqc,ncdhw->nqdhw', mask_embed, mask_features
        )

        # Build attention mask for the next decoder layer
        # Interpolate outputs_mask to target size
        attn_mask = F.interpolate(
            outputs_mask,
            size=attn_mask_target_size,
            mode='trilinear',
            align_corners=False,
        )  # (N, Q, D_t, H_t, W_t)

        # Flatten spatial dimensions: (N, Q, D_t*H_t*W_t)
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        if self.mask_classification:
            return [
                {'pred_logits': a, 'pred_masks': b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{'pred_masks': b} for b in outputs_seg_masks[:-1]]
