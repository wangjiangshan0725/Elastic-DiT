import functools
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .general_modules import AdaLayerNorm, RMSNorm, TimestepEmbeddings

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False


def sample_gate_st(logit: torch.Tensor, tau: float = 5.0):
    """
    logit: [B]
    tau:   Gumbel-Sigmoid 温度
    返回:
        gate: [B]，前向严格 0/1，反向用 soft 概率做 straight-through
        prob: [B]，soft 概率，用来算 FLOPs loss / 统计
    """
    # g = -torch.log(-torch.log(torch.rand_like(logit) + 1e-9) + 1e-9)
    # y = (logit + g) / tau
    prob = torch.sigmoid(logit)           # (0,1)

    gate_hard = (prob > 0.5).float()  # 0/1

    # ST：前向 = gate_hard，反向梯度 ≈ prob
    gate = gate_hard.detach() + prob - prob.detach()
    return gate, prob


class RouteGate(nn.Module):
    def __init__(self, dim: int, gate_hidden=128):
        super().__init__()
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim), 
        )

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        # self.width_head = nn.Sequential(
        #     nn.Linear(dim, gate_hidden),
        #     nn.ReLU(),
        #     nn.Linear(gate_hidden, 4),
        # )
        self.width_head = nn.Linear(gate_hidden, 4)
    
    def post_init(self, init_open_p=0.7, w_std=1e-3):
        """
        init_open_p: 初始开门概率（sigmoid 后），>=0.5
        w_std: 让 head 的权重很小，初期由 bias 主导，训练更稳
        """
        with torch.no_grad():
            # 1) trunk: Linear(dim->gate_hidden) 用常规初始化，方便训练
            nn.init.kaiming_uniform_(self.out[0].weight, a=math.sqrt(5))
            if self.out[0].bias is not None:
                self.out[0].bias.zero_()

            # 2) gate head: 小权重 + 正 bias，确保一开始概率 > 0.5
            nn.init.normal_(self.out[2].weight, mean=0.0, std=w_std)
            # logit(p) = log(p/(1-p))，p>0.5 => bias>0
            bias = math.log(init_open_p / (1.0 - init_open_p))
            if self.out[2].bias is not None:
                self.out[2].bias.fill_(bias)

            # 3) width head: 小权重 + 你原来的 bias 设定（保持初期偏好）
            nn.init.normal_(self.width_head.weight, mean=0.0, std=w_std)
            self.width_head.bias.zero_()
            self.width_head.bias[3] = 2.0

    def post_init_width(self,init_open_p=0.9, w_std=1e-3):
        with torch.no_grad():
            nn.init.normal_(self.width_head.weight, mean=0.0, std=w_std)
            self.width_head.bias.zero_()
            self.width_head.bias[3] = 2.0



    def forward(self, image, temb):
        img_mod_attn = self.img_mod(temb)#.chunk(2, dim=-1)  # [B, 3*dim] each
        img_normed = self.norm(image)
        img_modulated, _ = self._modulate(img_normed, img_mod_attn)

        # shared trunk: Linear(dim->gate_hidden) + ReLU
        # print("img_modulated.dtype:",img_modulated.dtype,"--self.out[0].dtype:",self.out[0].weight.dtype)
        # h = self.out[1](self.out[0](img_modulated.to(torch.bfloat16)))          # [B, L, gate_hidden]
        h = self.out[1](self.out[0](img_modulated))          # [B, L, gate_hidden]

        # original scalar gate logit head (kept exactly the same behavior)
        gate_logit = self.out[2](h).mean(dim=1)               # [B, 1]

        # new width head: gate_hidden -> 4 logits (for your linear_prob)
        width_logits = self.width_head(h).mean(dim=1)         # [B, 4]

        return gate_logit, width_logits

    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            # Assuming mod_params batch dim is 2*actual_batch (chunked into 2 parts)
            # So shift, scale, gate have shape [2*actual_batch, d]
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]  # each: [actual_batch, d]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            # index: [b, l] where b is actual batch size
            # Expand to [b, l, 1] to match feature dimension
            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            # Expand chunks to [b, 1, d] then broadcast to [b, l, d]
            shift_0_exp = shift_0.unsqueeze(1)  # [b, 1, d]
            shift_1_exp = shift_1.unsqueeze(1)  # [b, 1, d]
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            # Use torch.where to select based on index
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return (x * (1 + scale_result) + shift_result), gate_result
        

def qwen_image_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask = None, enable_fp8_attention: bool = False):
    if FLASH_ATTN_3_AVAILABLE and attention_mask is None:
        if not enable_fp8_attention:
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            if isinstance(x, tuple):
                x = x[0]
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
        else:
            origin_dtype = q.dtype
            q_std, k_std, v_std = q.std(), k.std(), v.std()
            q, k, v = (q / q_std).to(torch.float8_e4m3fn), (k / k_std).to(torch.float8_e4m3fn), (v / v_std).to(torch.float8_e4m3fn)
            q = rearrange(q, "b n s d -> b s n d", n=num_heads)
            k = rearrange(k, "b n s d -> b s n d", n=num_heads)
            v = rearrange(v, "b n s d -> b s n d", n=num_heads)
            x = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=q_std * k_std / math.sqrt(q.size(-1)))
            if isinstance(x, tuple):
                x = x[0]
            x = x.to(origin_dtype) * v_std
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


class ApproximateGELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)

def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.rope_cache = {}
        self.scale_rope = scale_rope
        
    def rope_params(self, index, dim, theta=10000):
        """
            Args:
                index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim))
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs


    def _expand_pos_freqs_if_needed(self, video_fhw, txt_seq_lens):
        if isinstance(video_fhw, list):
            video_fhw = tuple(max([i[j] for i in video_fhw]) for j in range(3))
        _, height, width = video_fhw
        if self.scale_rope:
            max_vid_index = max(height // 2, width // 2)
        else:
            max_vid_index = max(height, width)
        required_len = max_vid_index + max(txt_seq_lens)
        cur_max_len = self.pos_freqs.shape[0]
        if required_len <= cur_max_len:
            return

        new_max_len = math.ceil(required_len / 512) * 512
        pos_index = torch.arange(new_max_len)
        neg_index = torch.arange(new_max_len).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([
            self.rope_params(pos_index, self.axes_dim[0], self.theta),
            self.rope_params(pos_index, self.axes_dim[1], self.theta),
            self.rope_params(pos_index, self.axes_dim[2], self.theta),
        ], dim=1)
        self.neg_freqs = torch.cat([
            self.rope_params(neg_index, self.axes_dim[0], self.theta),
            self.rope_params(neg_index, self.axes_dim[1], self.theta),
            self.rope_params(neg_index, self.axes_dim[2], self.theta),
        ], dim=1)
        return


    def forward(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone().contiguous()
            vid_freqs.append(self.rope_cache[rope_key])

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs


    def forward_sampling(self, video_fhw, txt_seq_lens, device):
        self._expand_pos_freqs_if_needed(video_fhw, txt_seq_lens)
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if idx > 0 and f"{0}_{height}_{width}" not in self.rope_cache:
                frame_0, height_0, width_0 = video_fhw[0]

                rope_key_0 = f"0_{height_0}_{width_0}"
                spatial_freqs_0 = self.rope_cache[rope_key_0].reshape(frame_0, height_0, width_0, -1)
                h_indices = torch.linspace(0, height_0 - 1, height).long()
                w_indices = torch.linspace(0, width_0 - 1, width).long()
                h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
                sampled_rope = spatial_freqs_0[:, h_grid, w_grid, :]

                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                sampled_rope[:, :, :, :freqs_frame.shape[-1]] = freqs_frame

                seq_lens = frame * height * width
                self.rope_cache[rope_key] = sampled_rope.reshape(seq_lens, -1).clone()
            if rope_key not in self.rope_cache:
                seq_lens = frame * height * width
                freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
                freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
                if self.scale_rope:
                    freqs_height = torch.cat(
                        [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
                    )
                    freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
                    freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)

                else:
                    freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
                    freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

                freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
                self.rope_cache[rope_key] = freqs.clone()
            vid_freqs.append(self.rope_cache[rope_key].contiguous())

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs



class QwenEmbedLayer3DRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        video_fhw = [video_fhw]
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        layer_num = len(video_fhw) - 1
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            if idx != layer_num:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            else:
                ### For the condition image, we set the layer index to -1
                video_freq = self._compute_condition_freqs(frame, height, width)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_vid_index = max(max_vid_index, layer_num)
        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    @functools.lru_cache(maxsize=None)
    def _compute_condition_freqs(self, frame, height, width):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_neg[0][-1:].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(ApproximateGELU(dim, inner_dim))
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states, None


class QwenFeedForward_adaptive(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList([])
        self.net.append(ApproximateGELU(dim, inner_dim))  # fc1+act (assumed)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))    # fc2

        # ---- adaptive width: 4 prefix masks (H/4, H/2, 3H/4, H) ----
        h = inner_dim
        h1 = h // 4
        h2 = (2 * h) // 4
        h3 = (3 * h) // 4
        h4 = h

        masks = torch.zeros(4, h, dtype=torch.float32)
        masks[0, :h1] = 1.0
        masks[1, :h2] = 1.0
        masks[2, :h3] = 1.0
        masks[3, :h4] = 1.0
        self.register_buffer("mlp_prefix_masks", masks, persistent=False)

    def forward(self, hidden_states: torch.Tensor, logit_width=None, training=False, *args, **kwargs):
        """
        hidden_states: [B,L,C] 或 [B,C]
        logit_width:  None / [4] / [B,4] / [B,L,4]
        training:     True 时走原始 ST + mask 逻辑；False 时走“真正裁剪宽度”的推理分支
        返回:
            hidden_states_out: 与原实现完全等价的输出
            p_soft: [B,4]（若 logit_width 不为 None），否则 None
        """
        B = hidden_states.shape[0]
        p_soft = None

        # 如果没有提供 logit_width，就退化成普通 FFN
        if logit_width is None:
            hidden_states = self.net[0](hidden_states)  # Approx GELU
            hidden_states = self.net[1](hidden_states)
            hidden_states = self.net[2](hidden_states)
            return hidden_states, p_soft

        # -------- 先统一处理 logit_width，得到 p_soft --------
        logits = logit_width

        if logits.dim() == 1:
            # [4] -> [B,4]
            logits = logits.unsqueeze(0).expand(B, -1)
        elif logits.dim() == 3:
            # [B,L,4] -> [B,4]
            logits = logits.mean(dim=1)
        # 否则假设 [B,4]

        p_soft = torch.softmax(logits, dim=-1)      # [B,4]
        hard_idx = p_soft.argmax(dim=-1)           # [B]

        # prefix masks: [4,H]
        masks = self.mlp_prefix_masks.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        H = masks.shape[1]  # inner_dim

        # =============== 训练分支：保持原始逻辑不变 ===============
        if self.training:
            # ST：前向用 m_hard，反向梯度走 m_soft
            m_soft = p_soft @ masks            # [B,H]
            p_hard = F.one_hot(hard_idx, num_classes=4).to(
                dtype=p_soft.dtype,
                device=p_soft.device,
            )                                  # [B,4]
            m_hard = p_hard @ masks            # [B,H]
            m = (m_hard - m_soft).detach() + m_soft   # 前向=hard，反向=soft

            # 1) fc1 + act
            hidden_states = self.net[0](hidden_states)   # [..., H]

            # 2) mask
            if hidden_states.dim() == 3:      # [B,L,H]
                hidden_states = hidden_states * m[:, None, :]
            elif hidden_states.dim() == 2:    # [B,H]
                hidden_states = hidden_states * m
            else:
                # 泛化情况：在 batch 维做广播
                view_shape = (B,) + (1,) * (hidden_states.dim() - 2) + (H,)
                hidden_states = hidden_states * m.view(*view_shape)

            # 3) dropout + fc2
            hidden_states = self.net[1](hidden_states)
            hidden_states = self.net[2](hidden_states)
            return hidden_states, p_soft

        # =============== 推理分支：真正裁剪宽度 ===============
        # 此时 training == False，一般 model.eval()，dropout 也不会起作用

        # 取出 fc1 / fc2 权重（ApproximateGELU 内部的 Linear）
        proj: nn.Linear = self.net[0].proj          # dim -> H
        W1 = proj.weight                            # [H, C_in]
        b1 = proj.bias                              # [H] or None

        fc2: nn.Linear = self.net[2]                # H -> dim_out
        W2 = fc2.weight                             # [dim_out, H]
        b2 = fc2.bias                               # [dim_out]

        # 输出张量，形状和原 FFN 一致
        out_shape = hidden_states.shape[:-1] + (W2.shape[0],)
        out = hidden_states.new_empty(out_shape)    # [..., dim_out]

        # 预先算好每个宽度档位对应的前缀长度 k_i
        # mlp_prefix_masks 是 0/1，sum 即为通道数
        prefix_lengths = self.mlp_prefix_masks.sum(dim=1).to(torch.int64)  # [4]

        # 对每个宽度档位分别做子矩阵 Linear
        for width_idx in range(4):
            k = prefix_lengths[width_idx].item()
            if k <= 0:
                continue

            # 找出本 batch 中选择这一档宽度的样本
            mask_b = (hard_idx == width_idx)          # [B]
            if not mask_b.any():
                continue

            idx = mask_b.nonzero(as_tuple=True)[0]    # [B_group]
            x_group = hidden_states[idx]              # [B_group, ..., C_in]

            # --- fc1_k: x_group @ W1[:k, :].T + b1[:k] ---
            W1_k = W1[:k, :]                          # [k, C_in]
            b1_k = b1[:k] if b1 is not None else None # [k] or None

            h = F.linear(x_group, W1_k, b1_k)         # [B_group, ..., k]
            # Approximate GELU: x * sigmoid(1.702 * x)
            h = h * torch.sigmoid(1.702 * h)

            # --- fc2_k: h @ W2[:, :k].T + b2 ---
            W2_k = W2[:, :k]                          # [dim_out, k]
            out_group = F.linear(h, W2_k, b2)         # [B_group, ..., dim_out]

            # 写回对应样本的位置
            out[idx] = out_group

        # dropout 在 eval 模式下本来就是恒等，这里可以不调 self.net[1]，省掉一次调用
        # 如果你强迫症想 100% 路径一致，也可以在这里加一句：
        # out = self.net[1](out)

        return out, p_soft

class QwenDoubleStreamAttention(nn.Module):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        joint_attn_out = qwen_image_flash_attention(joint_q, joint_k, joint_v, num_heads=joint_q.shape[1], attention_mask=attention_mask, enable_fp8_attention=enable_fp8_attention).to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_attention_heads: int, 
        attention_head_dim: int, 
        eps: float = 1e-6,
    ):    
        super().__init__()
        
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim), 
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenFeedForward_adaptive(dim=dim, dim_out=dim)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True), 
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = QwenFeedForward(dim=dim, dim_out=dim)

        # cache_1 对应“同一 timestep 下第一次调用”的分支
        # cache_2 对应“同一 timestep 下第二次调用”的分支
        self.cache_1 = dict()
        self.cache_2 = dict()

        # 记录当前 block 内的 CFG 调用状态
        self._cache_state = {
            "last_step_key": None,  # 上一次的 timestep 标量 key
            "call_count": 0,        # 当前这个 timestep 下已经是第几次调用
            "step_index": -1,       # 已经历的 timestep 个数，从 0 开始计数
        }

    
    def _modulate(self, x, mod_params, index=None):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if index is not None:
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]

            index_expanded = index.unsqueeze(-1)  # [b, l, 1]

            shift_0_exp = shift_0.unsqueeze(1)
            shift_1_exp = shift_1.unsqueeze(1)
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)

            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)

        return x * (1 + scale_result) + shift_result, gate_result
    
    def forward(
        self,
        image: torch.Tensor,  
        text: torch.Tensor,
        temb: torch.Tensor, 
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        enable_fp8_attention: bool = False,
        modulate_index: Optional[List[int]] = None,
        gate_mlp = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = image.shape[0]

        # gate_mlp 用 text+image 做决策
        logit, logit_width = gate_mlp(image=torch.cat([text, image], dim=1), temb=temb)

        # ---------------------- 训练阶段：保持原逻辑不变 ----------------------
        if self.training:
            gate, prob = sample_gate_st(logit, tau=5.0)  # gate: 硬 0/1, prob: soft
            image_old = image
            text_old = text

            # if gate.item() == 0:
            #     p_soft = None
            #     return text, image, prob, p_soft 

            img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
            if modulate_index is not None:
                temb_txt = torch.chunk(temb, 2, dim=0)[0]
            else:
                temb_txt = temb
            txt_mod_attn, txt_mod_mlp = self.txt_mod(temb_txt).chunk(2, dim=-1)

            img_normed = self.img_norm1(image)
            img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

            txt_normed = self.txt_norm1(text)
            txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

            img_attn_out, txt_attn_out = self.attn(
                image=img_modulated,
                text=txt_modulated,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                enable_fp8_attention=enable_fp8_attention,
            )
            
            image = image + img_gate * img_attn_out
            text = text + txt_gate * txt_attn_out

            img_normed_2 = self.img_norm2(image)
            img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

            txt_normed_2 = self.txt_norm2(text)
            txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

            img_mlp_out, p_soft = self.img_mlp(img_modulated_2, logit_width=logit_width)
            txt_mlp_out, _ = self.txt_mlp(txt_modulated_2)

            image = image + img_gate_2 * img_mlp_out
            text = text + txt_gate_2 * txt_mlp_out

            gate_view = gate.view(B, 1, 1).to(image.dtype)
            image = image_old + gate_view * (image - image_old)
            text = text_old + gate_view * (text - text_old)

            return text, image, prob, p_soft

        # ---------------------- 推理阶段：block cache ----------------------
        prob = torch.sigmoid(logit)  # [B,1] or [B]
        prob_scalar = float(prob.mean().detach().item())
        # import pdb;pdb.set_trace()
        image_old = image
        text_old = text
        p_soft = None

        #--------------------------------
        # prob < 0.5：直接跳过
        if prob_scalar < 0.5:
            return text, image, prob, p_soft


        # —— 用 temb 决定这是“哪个 timestep”，并考虑浮点误差 ——
        temb_vec = temb[0].detach() if temb.dim() > 1 else temb.detach()
        step_key = float(temb_vec.mean().item())
        step_key_rounded = round(step_key, 6)

        state = self._cache_state
        if state["last_step_key"] is None or state["last_step_key"] != step_key_rounded:
            # 新的 timestep：重置该 timestep 的调用计数，并递增 step_index
            state["last_step_key"] = step_key_rounded
            state["call_count"] = 0
            state["step_index"] = state.get("step_index", -1) + 1

        call_idx = state["call_count"]
        state["call_count"] += 1

        # 是否启用 cache：前 5 个 timestep 固定不做 cache
        # step_index: 0,1,2,3,4 这五个步不 cache，从第 5 个开始 (index>=5) 才启用
        step_index = state["step_index"]
        
        # 8000 ckpt目前最快的
        use_cache = step_index >= 15

        use_cache = False

        cache_dict = self.cache_1 if (call_idx % 2 == 0) else self.cache_2
        
        # 8000 ckpt目前最快的
        # max_use = getattr(self, "cache_max_use", 10)
        max_use = getattr(self, "cache_max_use", 10)

        # ---------------- prob 在 0.5 ~ 0.505 之间：尝试用 cache ----------------

        # 8000 ckpt目前最快的s
        if use_cache and (0.5 <= prob_scalar <= 0.515):
        # if use_cache and (0.5 <= prob_scalar <= 0.51):
            if (
                "delta_image" in cache_dict
                and cache_dict.get("use_count", 0) < max_use
                and cache_dict["delta_image"].shape == image_old.shape
                and cache_dict["delta_text"].shape == text_old.shape
            ):
                delta_image = cache_dict["delta_image"]
                delta_text = cache_dict["delta_text"]
                cached_p_soft = cache_dict.get("p_soft", None)

                image = image_old + delta_image
                text = text_old + delta_text
                p_soft = cached_p_soft

                cache_dict["use_count"] = cache_dict.get("use_count", 0) + 1
                return text, image, prob, p_soft
            # 否则：没有可用 cache，下面完整 forward（并在 use_cache=True 时更新 cache）

        # ---------------- prob > 0.505 或无可用 cache：完整 forward ----------------
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
        if modulate_index is not None:
            temb_txt = torch.chunk(temb, 2, dim=0)[0]
        else:
            temb_txt = temb
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb_txt).chunk(2, dim=-1)

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
        )
        
        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out, p_soft = self.img_mlp(img_modulated_2, logit_width=logit_width)
        txt_mlp_out, _ = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        # 推理阶段这里不再做 ST skip（原来 gate>0.5 时 gate_view=1，是恒等）
        # 现在 block 输出就是 image / text

        # 若当前 timestep 还没到第 5 个，不做 cache，直接返回
        if not use_cache:
            return text, image, prob, p_soft

        # 更新当前分支的 cache：存输出与输入的差，以及 p_soft
        delta_image = (image - image_old).detach()
        delta_text = (text - text_old).detach()

        cache_dict.clear()
        cache_dict["delta_image"] = delta_image
        cache_dict["delta_text"] = delta_text
        cache_dict["p_soft"] = p_soft.detach() if p_soft is not None else None
        cache_dict["use_count"] = 0

        return text, image, prob, p_soft
    
    
    # def forward(
    #     self,
    #     image: torch.Tensor,  
    #     text: torch.Tensor,
    #     temb: torch.Tensor, 
    #     image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     enable_fp8_attention: bool = False,
    #     modulate_index: Optional[List[int]] = None,
    #     gate_mlp = None,
    #     training: bool = True,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:

    #     B = image.shape[0]

    #     # gate_mlp 用 text+image 做决策
    #     logit, logit_width = gate_mlp(image=torch.cat([text, image], dim=1), temb=temb)

    #     # ---------------------- 训练阶段：保持原逻辑不变 ----------------------
    #     if self.training:
    #         gate, prob = sample_gate_st(logit, tau=5.0)  # gate: 硬 0/1, prob: soft
    #         image_old = image
    #         text_old = text

    #         # if gate.item() == 0:
    #         #     p_soft = None
    #         #     return text, image, prob, p_soft 

    #         img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
    #         if modulate_index is not None:
    #             temb_txt = torch.chunk(temb, 2, dim=0)[0]
    #         else:
    #             temb_txt = temb
    #         txt_mod_attn, txt_mod_mlp = self.txt_mod(temb_txt).chunk(2, dim=-1)

    #         img_normed = self.img_norm1(image)
    #         img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

    #         txt_normed = self.txt_norm1(text)
    #         txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

    #         img_attn_out, txt_attn_out = self.attn(
    #             image=img_modulated,
    #             text=txt_modulated,
    #             image_rotary_emb=image_rotary_emb,
    #             attention_mask=attention_mask,
    #             enable_fp8_attention=enable_fp8_attention,
    #         )
            
    #         image = image + img_gate * img_attn_out
    #         text = text + txt_gate * txt_attn_out

    #         img_normed_2 = self.img_norm2(image)
    #         img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

    #         txt_normed_2 = self.txt_norm2(text)
    #         txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

    #         img_mlp_out, p_soft = self.img_mlp(img_modulated_2, logit_width=logit_width)
    #         txt_mlp_out, _ = self.txt_mlp(txt_modulated_2)

    #         image = image + img_gate_2 * img_mlp_out
    #         text = text + txt_gate_2 * txt_mlp_out

    #         gate_view = gate.view(B, 1, 1).to(image.dtype)
    #         image = image_old + gate_view * (image - image_old)
    #         text = text_old + gate_view * (text - text_old)

    #         return text, image, prob, p_soft

    #     # ---------------------- 推理阶段：block cache ----------------------
    #     prob = torch.sigmoid(logit)  # [B,1] or [B]
    #     prob_scalar = float(prob.mean().detach().item())

    #     image_old = image
    #     text_old = text
    #     p_soft = None

    #     # —— 用 temb 决定这是“哪个 timestep”，并考虑浮点误差 ——
    #     temb_vec = temb[0].detach() if temb.dim() > 1 else temb.detach()
    #     step_key = float(temb_vec.mean().item())
    #     step_key_rounded = round(step_key, 6)

    #     state = self._cache_state
    #     if state["last_step_key"] is None or state["last_step_key"] != step_key_rounded:
    #         # 新的 timestep：重置该 timestep 的调用计数，并递增 step_index
    #         state["last_step_key"] = step_key_rounded
    #         state["call_count"] = 0
    #         state["step_index"] = state.get("step_index", -1) + 1

    #     call_idx = state["call_count"]
    #     state["call_count"] += 1

    #     # 是否启用 cache（你现在是从第 15 个 step 开始启用）
    #     step_index = state["step_index"]
    #     use_cache = step_index >= 15

    #     # 第一次调用 -> cache_1；第二次调用 -> cache_2；再多也按 0/1 轮流
    #     cache_dict = self.cache_1 if (call_idx % 2 == 0) else self.cache_2
    #     max_use = getattr(self, "cache_max_use", 7)

    #     # --------- 新逻辑：prob < 0.5 也先查 cache，没有再真正跳过 ----------
    #     if prob_scalar < 0.5:
    #         if (
    #             use_cache
    #             and "delta_image" in cache_dict
    #             and cache_dict.get("use_count", 0) < max_use
    #             and cache_dict["delta_image"].shape == image_old.shape
    #             and cache_dict["delta_text"].shape == text_old.shape
    #         ):
    #             # 低概率但有 cache，直接用 cache 的 delta
    #             delta_image = cache_dict["delta_image"]
    #             delta_text = cache_dict["delta_text"]
    #             cached_p_soft = cache_dict.get("p_soft", None)

    #             image = image_old + delta_image
    #             text = text_old + delta_text
    #             p_soft = cached_p_soft

    #             cache_dict["use_count"] = cache_dict.get("use_count", 0) + 1
    #             return text, image, prob, p_soft

    #         # 没有可用 cache，才真正跳过（保持原 feature）
    #         return text, image, prob, p_soft
    #     # -------------------------------------------------------------------

    #     # ---------------- prob 在 0.5 ~ 0.508 之间：尝试用 cache ----------------
    #     if use_cache and (0.5 <= prob_scalar <= 0.515):
    #         if (
    #             "delta_image" in cache_dict
    #             and cache_dict.get("use_count", 0) < max_use
    #             and cache_dict["delta_image"].shape == image_old.shape
    #             and cache_dict["delta_text"].shape == text_old.shape
    #         ):
    #             delta_image = cache_dict["delta_image"]
    #             delta_text = cache_dict["delta_text"]
    #             cached_p_soft = cache_dict.get("p_soft", None)

    #             image = image_old + delta_image
    #             text = text_old + delta_text
    #             p_soft = cached_p_soft

    #             cache_dict["use_count"] = cache_dict.get("use_count", 0) + 1
    #             return text, image, prob, p_soft
    #         # 否则：没有可用 cache，下面完整 forward（并在 use_cache=True 时更新 cache）

    #     # ---------------- prob > 0.508 或无可用 cache：完整 forward ----------------
    #     img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
    #     if modulate_index is not None:
    #         temb_txt = torch.chunk(temb, 2, dim=0)[0]
    #     else:
    #         temb_txt = temb
    #     txt_mod_attn, txt_mod_mlp = self.txt_mod(temb_txt).chunk(2, dim=-1)

    #     img_normed = self.img_norm1(image)
    #     img_modulated, img_gate = self._modulate(img_normed, img_mod_attn, index=modulate_index)

    #     txt_normed = self.txt_norm1(text)
    #     txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)

    #     img_attn_out, txt_attn_out = self.attn(
    #         image=img_modulated,
    #         text=txt_modulated,
    #         image_rotary_emb=image_rotary_emb,
    #         attention_mask=attention_mask,
    #         enable_fp8_attention=enable_fp8_attention,
    #     )
        
    #     image = image + img_gate * img_attn_out
    #     text = text + txt_gate * txt_attn_out

    #     img_normed_2 = self.img_norm2(image)
    #     img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp, index=modulate_index)

    #     txt_normed_2 = self.txt_norm2(text)
    #     txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

    #     img_mlp_out, p_soft = self.img_mlp(img_modulated_2, logit_width=logit_width)
    #     txt_mlp_out, _ = self.txt_mlp(txt_modulated_2)

    #     image = image + img_gate_2 * img_mlp_out
    #     text = text + txt_gate_2 * txt_mlp_out

    #     # 推理阶段这里不再做 ST skip（原来 gate>0.5 时 gate_view=1，是恒等）
    #     # 现在 block 输出就是 image / text

    #     # 若当前 timestep 未启用 cache，直接返回
    #     if not use_cache:
    #         return text, image, prob, p_soft

    #     # 更新当前分支的 cache：存输出与输入的差，以及 p_soft
    #     delta_image = (image - image_old).detach()
    #     delta_text = (text - text_old).detach()

    #     cache_dict.clear()
    #     cache_dict["delta_image"] = delta_image
    #     cache_dict["delta_text"] = delta_text
    #     cache_dict["p_soft"] = p_soft.detach() if p_soft is not None else None
    #     cache_dict["use_count"] = 0

    #     return text, image, prob, p_soft


class QwenImageDiT(torch.nn.Module):
    def __init__(
        self,
        num_layers: int = 60,
        use_layer3d_rope: bool = False,
        use_additional_t_cond: bool = False,
    ):
        super().__init__()

        if not use_layer3d_rope:
            self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16,56,56], scale_rope=True)
        else:
            self.pos_embed = QwenEmbedLayer3DRope(theta=10000, axes_dim=[16,56,56], scale_rope=True)

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=False, use_additional_t_cond=use_additional_t_cond)
        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)
        # self.hidden_size = 128
        self.tau = 5.0

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.gate_mlps = nn.ModuleList(
            [
                RouteGate(3072)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)


    def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask, entity_masks, height, width, image, img_shapes):
        # prompt_emb
        all_prompt_emb = entity_prompt_emb + [prompt_emb]
        all_prompt_emb = [self.txt_in(self.txt_norm(local_prompt_emb)) for local_prompt_emb in all_prompt_emb]
        all_prompt_emb = torch.cat(all_prompt_emb, dim=1)

        # image_rotary_emb
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        entity_seq_lens = [emb_mask.sum(dim=1).tolist() for emb_mask in entity_prompt_emb_mask]
        entity_rotary_emb = [self.pos_embed(img_shapes, entity_seq_len, device=latents.device)[1] for entity_seq_len in entity_seq_lens]
        txt_rotary_emb = torch.cat(entity_rotary_emb + [image_rotary_emb[1]], dim=0)
        image_rotary_emb = (image_rotary_emb[0], txt_rotary_emb)

        # attention_mask
        repeat_dim = latents.shape[1]
        max_masks = entity_masks.shape[1]
        entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
        entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
        global_mask = torch.ones_like(entity_masks[0]).to(device=latents.device, dtype=latents.dtype)
        entity_masks = entity_masks + [global_mask]

        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        seq_lens = [mask_.sum(dim=1).item() for mask_ in entity_prompt_emb_mask] + [prompt_emb_mask.sum(dim=1).item()]
        total_seq_len = sum(seq_lens) + image.shape[1]
        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(entity_masks[i], "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
            patched_masks.append(patched_mask)
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(device=entity_masks[0].device)

        # prompt-image attention mask
        image_start = sum(seq_lens)
        image_end = total_seq_len
        cumsum = [0]
        single_image_seq = image_end - image_start
        for length in seq_lens:
            cumsum.append(cumsum[-1] + length)
        for i in range(N):
            prompt_start = cumsum[i]
            prompt_end = cumsum[i+1]
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, seq_lens[i], 1)
            # repeat image mask to match the single image sequence length
            repeat_time = single_image_seq // image_mask.shape[-1]
            image_mask = image_mask.repeat(1, 1, repeat_time)
            # prompt update with image
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # image update with prompt
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        # prompt-prompt attention mask, let the prompt tokens not attend to each other
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                start_i, end_i = cumsum[i], cumsum[i+1]
                start_j, end_j = cumsum[j], cumsum[j+1]
                attention_mask[:, start_i:end_i, start_j:end_j] = False

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        attention_mask = attention_mask.to(device=latents.device, dtype=latents.dtype).unsqueeze(1)

        return all_prompt_emb, image_rotary_emb, attention_mask


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
        training=True,
    ):
        return None
        # img_shapes = [(latents.shape[0], latents.shape[2]//2, latents.shape[3]//2)]
        # txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        
        # image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2)
        # image = self.img_in(image)
        # text = self.txt_in(self.txt_norm(prompt_emb))

        # conditioning = self.time_text_embed(timestep, image.dtype)

        # image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        
        # probs_list = []
        # for block, gate_mlp in zip(self.transformer_blocks, self.gate_mlps):

        #     B = image.shape[0]
        #     logit, logit_width = gate_mlp(image=image, temb=conditioning)
        #     if training:
        #         gate, prob = sample_gate_st(logit, tau=self.tau)#, tau=self.tau_double)   # gate: 硬 0/1
        #     else:
        #         prob = torch.sigmoid(logit)
        #         gate = (prob > 0.5).to(image.dtype)

        #     text_new, image_new = block(
        #         image=image,
        #         text=text,
        #         temb=conditioning,
        #         image_rotary_emb=image_rotary_emb,
        #     )

        #     gate_view = gate.view(B, 1, 1)  
        #     image = image + gate_view * (image_new - image)
        #     text = text + gate_view * (text_new - text)

        #     probs_list.append(prob)
        
        # image = self.norm_out(image, conditioning)
        # image = self.proj_out(image)
        
        # latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2)
        # # dasdsada
        # print(probs_list)
        # return image, probs_list
