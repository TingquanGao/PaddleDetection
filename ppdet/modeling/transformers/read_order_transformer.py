"""This file contains the ReadOrderTransformer module, which is designed to be
structurally consistent with LayoutLMv3 for reading order prediction.
"""

from __future__ import absolute_import, division, print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Optional, Union

from ..initializer import linear_init_
from .utils import GlobalPointerPD

# =========================
# Utils for Relation-DETR
# =========================
def box_rel_encoding_pd(src_boxes: paddle.Tensor, tgt_boxes: paddle.Tensor=None, eps: float=1e-5):
    """
    src_boxes: [B, N1, 4] (x, y, w, h)
    tgt_boxes: [B, N2, 4]
    return:    [B, N1, N2, 4] with [log(|dx|/w_i + 1), log(|dy|/h_i + 1), log(w_i/w_j), log(h_i/h_j)]
    """
    if tgt_boxes is None:
        tgt_boxes = src_boxes
    assert src_boxes.shape[-1] == 4 and tgt_boxes.shape[-1] == 4
    xy1, wh1 = src_boxes[..., :2], src_boxes[..., 2:]
    xy2, wh2 = tgt_boxes[..., :2], tgt_boxes[..., 2:]
    delta_xy = paddle.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))                    # [B,N1,N2,2]
    delta_xy = paddle.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)               # [B,N1,N2,2]
    delta_wh = paddle.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))    # [B,N1,N2,2]
    pos = paddle.concat([delta_xy, delta_wh], axis=-1)                              # [B,N1,N2,4]
    return pos

def get_sine_pos_embed_pd(x: paddle.Tensor,
                          num_pos_feats: int,
                          temperature: float = 10000.0,
                          scale: float = 100.0,
                          exchange_xy: bool = False):
    """
    x: [B, N1, N2, C] (C=4 for relation)
    returns: [B, N1, N2, C * num_pos_feats]  (sin/cos 多频)
    """
    if exchange_xy and x.shape[-1] >= 2:
        x = paddle.stack([x[..., 1], x[..., 0], *([x[..., i] for i in range(2, x.shape[-1])])], axis=-1)

    half = num_pos_feats // 2
    dim_t = temperature ** (2 * paddle.arange(half, dtype='float32') / half)  # [half]

    def _encode(t: paddle.Tensor):
        t = t * scale
        t = t.unsqueeze(-1) / dim_t  # [B,N1,N2,half]
        sin = paddle.sin(t)
        cos = paddle.cos(t)
        return paddle.concat([sin, cos], axis=-1)  # [B,N1,N2,num_pos_feats]

    embs = [_encode(x[..., i]) for i in range(x.shape[-1])]
    out = paddle.concat(embs, axis=-1)  # [B,N1,N2, num_pos_feats * C]
    return out

class PositionRelationEmbeddingPD(nn.Layer):
    """
    Paddle 版 Relation-DETR 的几何关系偏置生成：
    (cx,cy,w,h) → 4维相对量 → 多频 sin/cos → 1x1 Conv → [B, H, L, L]
    """
    def __init__(self,
                 embed_dim: int,          # e.g., 16
                 num_heads: int,
                 temperature: float = 10000.0,
                 scale: float = 100.0):
        super().__init__()
        in_ch = embed_dim * 4            # 4个量，每个量 embed_dim 维
        self.pos_proj = nn.Conv2D(in_channels=in_ch, out_channels=num_heads, kernel_size=1)
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.scale = scale

    def forward(self, src_boxes: paddle.Tensor, tgt_boxes: paddle.Tensor=None):
        """
        src_boxes: [B, L, 4] (cx,cy,w,h)
        tgt_boxes: [B, L, 4] (默认同 src)
        return:    [B, H, L, L]
        """
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        with paddle.no_grad():
            rel = box_rel_encoding_pd(src_boxes, tgt_boxes)                                 # [B,L,L,4]
            pos = get_sine_pos_embed_pd(rel, num_pos_feats=self.embed_dim,
                                        temperature=self.temperature, scale=self.scale)     # [B,L,L,embed_dim*4]
            pos = pos.transpose([0, 3, 1, 2])                                               # [B,C,L,L]
        out = self.pos_proj(pos)                                                            # [B,H,L,L]
        return out

# A local MLP class to keep the module self-contained.
class MLP(nn.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# === Begin: LayoutLMv3 Components ===

class LayoutLMv3SelfAttention(nn.Layer):
    def __init__(self, hidden_size, num_attention_heads, dropout_prob,
                 has_relative_attention_bias, has_spatial_attention_bias):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})")

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias  # 保留字段，不再使用旧2D

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def cogview_attention(self, attention_scores, alpha=32):
        scaled_attention_scores = attention_scores / alpha
        max_value = paddle.max(scaled_attention_scores, axis=-1, keepdim=True)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(axis=-1)(new_attention_scores)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, rel_pos=None, rel_geom_bias=None):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 1D 相对位置（句位）保持除以 sqrt(d)（与你原逻辑一致）
        if self.has_relative_attention_bias and (rel_pos is not None):
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)
        # 几何关系偏置（Relation-DETR）：直接加，不再除以 sqrt(d)
        if rel_geom_bias is not None:
            # 保证 dtype 一致
            if rel_geom_bias.dtype != attention_scores.dtype:
                rel_geom_bias = rel_geom_bias.astype(attention_scores.dtype)
            attention_scores += rel_geom_bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.astype('float32')

        attention_probs = self.cogview_attention(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class LayoutLMv3SelfOutput(nn.Layer):
    def __init__(self, hidden_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LayoutLMv3Attention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttention(
            config['hidden_size'],
            config['num_attention_heads'],
            config['attention_probs_dropout_prob'],
            config['has_relative_attention_bias'],
            config['has_spatial_attention_bias']
        )
        self.output = LayoutLMv3SelfOutput(
            config['hidden_size'],
            config['layer_norm_eps'],
            config['hidden_dropout_prob']
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, rel_pos=None, rel_geom_bias=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask,
                                 output_attentions, rel_pos, rel_geom_bias)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

class LayoutLMv3Intermediate(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = getattr(F, hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class LayoutLMv3Output(nn.Layer):
    def __init__(self, hidden_size, intermediate_size, layer_norm_eps, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LayoutLMv3Layer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = LayoutLMv3Attention(config)
        self.intermediate = LayoutLMv3Intermediate(
            config['hidden_size'],
            config['intermediate_size'],
            config['hidden_act']
        )
        self.output = LayoutLMv3Output(
            config['hidden_size'],
            config['intermediate_size'],
            config['layer_norm_eps'],
            config['hidden_dropout_prob']
        )

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, rel_pos=None, rel_geom_bias=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask,
                                           output_attentions, rel_pos, rel_geom_bias)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs

class LayoutLMv3Encoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.LayerList([LayoutLMv3Layer(config) for _ in range(config['num_hidden_layers'])])
        self.has_relative_attention_bias = config['has_relative_attention_bias']

        # 1D 相对位置（保留）
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config['rel_pos_bins']
            self.max_rel_pos = config['max_rel_pos']
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config['num_attention_heads'], bias_attr=False)

        # Relation-DETR 几何偏置（新增）
        self.rel_bias_module = PositionRelationEmbeddingPD(
            embed_dim=16,  # 可调：8/16/32
            num_heads=config['num_attention_heads'],
            temperature=10000.0,
            scale=100.0
        )

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).astype("int64") * num_buckets
            n = paddle.abs(relative_position)
        else:
            n = paddle.maximum(-relative_position, paddle.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            paddle.log(n.astype("float32") / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).astype("int64")
        val_if_large = paddle.minimum(val_if_large, paddle.full_like(val_if_large, num_buckets - 1))
        ret += paddle.where(is_small, n, val_if_large)
        return ret

    def _cal_1d_pos_emb(self, position_ids):
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos = self.relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins, max_distance=self.max_rel_pos)
        with paddle.no_grad():
            rel_pos = (
                self.rel_pos_bias.weight                      # [n_heads, rel_pos_bins]
                .reshape([self.rel_pos_bins, -1])             # [rel_pos_bins, n_heads]
                [rel_pos]                                     # [B, L, L, n_heads]
                .transpose((0, 3, 1, 2))                      # [B, n_heads, L, L]
            )
        return rel_pos

    def _boxes_xyxy_to_cxcywh(self, bbox_xyxy: paddle.Tensor):
        # bbox_xyxy: [B, L, 4] (x1,y1,x2,y2)  —— 这里转为 (cx,cy,w,h)
        x1, y1, x2, y2 = bbox_xyxy[..., 0].astype('float32'), bbox_xyxy[..., 1].astype('float32'), \
                         bbox_xyxy[..., 2].astype('float32'), bbox_xyxy[..., 3].astype('float32')
        w = (x2 - x1).clip(min=1e-3)
        h = (y2 - y1).clip(min=1e-3)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        return paddle.stack([cx, cy, w, h], axis=-1)  # [B,L,4]

    def forward(self, hidden_states, bbox, attention_mask=None, head_mask=None,
                output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        B, L = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = paddle.arange(L, dtype="int64").expand((B, -1))

        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None

        # Relation bias：一次计算，全层复用
        boxes_cxcywh = self._boxes_xyxy_to_cxcywh(bbox)                # [B,L,4]
        rel_geom_bias = self.rel_bias_module(boxes_cxcywh)             # [B,H,L,L]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask,
                                         output_attentions, rel_pos, rel_geom_bias)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states, all_self_attentions


class LayoutLMv3TextEmbeddings(nn.Layer):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config['vocab_size'],
            config['hidden_size'],
            padding_idx=config['pad_token_id']
        )
        self.token_type_embeddings = nn.Embedding(
            config['type_vocab_size'],
            config['hidden_size']
        )

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], epsilon=config['layer_norm_eps'])
        self.dropout = nn.Dropout(p=config['hidden_dropout_prob'])

        self.position_ids = paddle.arange(config['max_position_embeddings']).expand((1, -1))
        self.position_ids.stop_gradient = True

        self.padding_idx = config['pad_token_id']
        self.position_embeddings = nn.Embedding(
            config['max_position_embeddings'],
            config['hidden_size'],
            padding_idx=self.padding_idx
        )

        self.x_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['coordinate_size'])
        self.y_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['coordinate_size'])
        self.h_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['shape_size'])
        self.w_position_embeddings = nn.Embedding(config['max_2d_position_embeddings'], config['shape_size'])

        self.spatial_embed_dim = 4 * config['coordinate_size'] + 2 * config['shape_size']  # 期望=1024
        self.spatial_proj = nn.Linear(self.spatial_embed_dim, config['hidden_size'])

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            bbox = paddle.clip(bbox, 0, 1023)  # 防越界
            left_position_embeddings  = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(
            paddle.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023)
        )
        w_position_embeddings = self.w_position_embeddings(
            paddle.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023)
        )

        spatial_position_embeddings = paddle.concat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            axis=-1,
        )
        return spatial_position_embeddings

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        mask = (input_ids != padding_idx).astype('int64')
        incremental_indices = paddle.cumsum(mask, axis=1) * mask
        return incremental_indices.astype('int64') + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.shape[:-1]
        sequence_length = input_shape[1]
        position_ids = paddle.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype='int64'
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(self, input_ids=None, bbox=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(
                    input_ids, self.padding_idx
                ).to(input_ids.place)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype='int64')

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)
        spatial_position_embeddings = self.spatial_proj(spatial_position_embeddings)

        ##跟输入序列的绝对位置都不要了
        embeddings += spatial_position_embeddings

        return embeddings


class LayoutLMv3ClassificationHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        classifier_dropout = config["hidden_dropout_prob"]
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config["hidden_size"], config["num_labels"])

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = paddle.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


from .deformable_transformer import MSDeformableAttention  # 项目里已有

class BoxGlobalAggregator(nn.Layer):
    def __init__(self, d_model, nhead=16, n_levels=3, n_points=4):
        super().__init__()
        self.proj_in  = nn.Linear(d_model, d_model)
        self.proj_val = nn.Linear(256, d_model)
        self.msda     = MSDeformableAttention(d_model, nhead, n_levels, n_points, 1.0)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, query_tokens, boxes_xyxy_norm,
                global_memory, spatial_shapes, level_start_index):
        bs, Lq, D = query_tokens.shape
        N = boxes_xyxy_norm.shape[1]
        x1, y1, x2, y2 = [boxes_xyxy_norm[:, :, i] for i in range(4)]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w  = (x2 - x1).clip(min=1e-6)
        h  = (y2 - y1).clip(min=1e-6)
        ref = paddle.stack([cx, cy, w, h], axis=-1)        # [bs, N, 4]
        n_levels = len(spatial_shapes)
        ref = ref.unsqueeze(2).tile([1, 1, n_levels, 1])   # [bs, N, L, 4]

        val = self.proj_val(global_memory)                 # [bs, Lv, D]
        q_boxes = self.proj_in(query_tokens[:, 1:N+1, :])  # [bs, N, D]
        out_ctx = self.msda(q_boxes, ref, val, spatial_shapes, level_start_index)  # [bs, N, D]
        out_ctx = self.proj_out(out_ctx)

        enriched = query_tokens.clone()
        enriched[:, 1:N+1, :] = enriched[:, 1:N+1, :] + out_ctx
        return enriched




class ReadOrderTransformer(nn.Layer):
    def __init__(self):
        super(ReadOrderTransformer, self).__init__()   
        self.config = {
            'hidden_size': 512,
            'num_attention_heads': 8,
            'attention_probs_dropout_prob': 0.1,   #修改
            'has_relative_attention_bias': False,
            'has_spatial_attention_bias': True,
            'layer_norm_eps': 1e-5,
            'hidden_dropout_prob': 0.1,  #修改
            'intermediate_size': 2048,
            'hidden_act': 'gelu',
            'num_hidden_layers':6,
            'rel_pos_bins': 32,
            'max_rel_pos': 128,
            'rel_2d_pos_bins': 64,
            'max_rel_2d_pos': 256,
            'num_labels': 510,
            'max_position_embeddings':514,
            "max_2d_position_embeddings": 1024,
            "type_vocab_size": 1,
            "vocab_size": 4,
            "pad_token_id": 1,
            "coordinate_size": 171,
            "shape_size": 170,
            "num_classes": 20,
        }

        self.embeddings = LayoutLMv3TextEmbeddings(self.config)
        self.label_embeddings = nn.Embedding(self.config["num_classes"], self.config['hidden_size'])
        self.label_features_projection = nn.Linear(self.config['hidden_size'], self.config['hidden_size'])
        self.visual_features_projection = nn.Linear(256, self.config['hidden_size'])

        # 全局视觉
        self.global_visual_proj = nn.Linear(256, self.config['hidden_size'])
        self.global_gate = nn.Sequential(
            nn.Linear(self.config['hidden_size'], self.config['hidden_size']),
            nn.Sigmoid()
        )

        self.global_agg = BoxGlobalAggregator(d_model=self.config['hidden_size'],
                                              nhead=self.config['num_attention_heads'],
                                              n_levels=3, n_points=4)

        self.encoder = LayoutLMv3Encoder(self.config)
        self.dropout = nn.Dropout(self.config['hidden_dropout_prob'])
        
        self.relative_head = GlobalPointerPD(
            hidden_size=self.config['hidden_size'],
            heads=1,
            head_size=64,
            use_rope=False,
            tril_mask=True,
            max_length=512
        )

    def forward(self, boxes_list, labels_list=None, 
                global_visual=None, global_memory=None, global_spatial_shapes=None, global_level_start_index=None):

        MAX_LEN = 510
        CLS_TOKEN_ID = 0
        UNK_TOKEN_ID = 3
        EOS_TOKEN_ID = 2

        batch_size = len(boxes_list)
        max_len = max(len(boxes) for boxes in boxes_list)
        

        input_ids_list, bbox_list, attention_mask_list = [], [], []
        for boxes in boxes_list:
            # 截断保护
            if len(boxes) > MAX_LEN:
                boxes = boxes[:MAX_LEN]
            seq_len = len(boxes)
            input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * seq_len + [EOS_TOKEN_ID]
            bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
            mask = [1] * (seq_len + 2)

            pad_len = max_len + 2 - len(input_ids)
            input_ids += [self.config['pad_token_id']] * pad_len
            bbox += [[0, 0, 0, 0]] * pad_len
            mask += [0] * pad_len

            input_ids_list.append(input_ids)
            bbox_list.append(bbox)
            attention_mask_list.append(mask)

        input_ids = paddle.to_tensor(input_ids_list, dtype='int64')
        bbox = paddle.to_tensor(bbox_list, dtype='int64')
        attention_mask = paddle.to_tensor(attention_mask_list, dtype='int64')
        attention_mask = attention_mask.unsqueeze([1, 2])  # [B, 1, 1, L]
        attention_mask = (1.0 - attention_mask) * -1e9

        # 文本+布局 embedding
        bbox_embedding = self.embeddings(input_ids=input_ids, bbox=bbox)

        # 类别特征
        final_label_features = paddle.zeros([batch_size, max_len + 2, self.config['hidden_size']], dtype='float32')
        if labels_list is not None:
            for i in range(batch_size):
                labels = labels_list[i]
                num_labels = labels.shape[0]
                if num_labels > 0:
                    label_embs_sample = self.label_embeddings(labels)                 # [N_i, H]
                    projected_label_features_sample = self.label_features_projection(label_embs_sample)  # [N_i, H]
                    final_label_features[i, 1:num_labels+1, :] = projected_label_features_sample.squeeze(1)

        final_embeddings = bbox_embedding  + final_label_features 

        # # 全局视觉注入
        # if global_visual is not None:
        #     g_proj = self.global_visual_proj(global_visual)           # [B, H]
        #     final_embeddings[:, 0, :] = final_embeddings[:, 0, :] + g_proj
        #     gamma = self.global_gate(g_proj)                          # [B, H] in (0,1)
        #     final_embeddings = final_embeddings + gamma.unsqueeze(1)

        final_embeddings = self.embeddings.LayerNorm(final_embeddings)
        final_embeddings = self.embeddings.dropout(final_embeddings)

        # # 归一化 boxes 到 [0,1]，供 MSDA
        # boxes_xyxy_norm_list = []
        # for boxes in boxes_list:
        #     if len(boxes) == 0:
        #         boxes_xyxy_norm_list.append([])
        #     else:
        #         arr = paddle.to_tensor(boxes, dtype='float32') / 1000.0
        #         boxes_xyxy_norm_list.append(arr)
        # max_len_b = max(len(b) for b in boxes_list)
        # padded_norm = paddle.zeros([batch_size, max_len_b, 4], 'float32')
        # for i, arr in enumerate(boxes_xyxy_norm_list):
        #     if len(arr) > 0: padded_norm[i, :arr.shape[0], :] = arr

        # if (global_memory is not None) and (global_spatial_shapes is not None):
        #     spatial_shapes = paddle.to_tensor(global_spatial_shapes)
        #     level_start_index = paddle.to_tensor(global_level_start_index)
        #     final_embeddings = self.global_agg(
        #         final_embeddings, padded_norm,
        #         global_memory, spatial_shapes, level_start_index
        #     )

        # 编码器
        encoder_output, _, _ = self.encoder(hidden_states=final_embeddings, bbox=bbox, attention_mask=attention_mask)

        # 相对排序 logits
        lengths = paddle.to_tensor([len(b) for b in boxes_list], dtype='int64')
        N_max = int(lengths.max().item())
        tok = encoder_output[:, 1:1+N_max, :]                               # [B, N, H]
        attn_1d = (paddle.arange(N_max)[None, :].tile([tok.shape[0], 1]) < lengths[:, None]).astype('float32')  # [B,N]
        logits_bh, mask_b = self.relative_head(tok, attn_1d)                   # [B,1,N,N], [B,1,N,N]
        read_order_logits = logits_bh[:, 0]                                  # [B,N,N]


        return read_order_logits