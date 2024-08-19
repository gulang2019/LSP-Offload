import math
from typing import Union
import transformers
import torch.nn as nn
import torch
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from typing import Optional, Tuple, Union
from torch.cuda.amp import autocast
from lora import *
from utils import matrix_analyzer
import inspect
# low-rank adaptor
class LoRALayer(nn.Module):
    def __init__(self, module: Union[torch.nn.Linear, transformers.pytorch_utils.Conv1D], rank: int, alpha = 8, split = 1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        if isinstance(module, nn.Linear):
            self.in_features = module.in_features
            self.out_features = module.out_features
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            self.in_features = module.weight.size(0)
            self.out_features = module.weight.size(1)
        self.linear = module
        self.alpha = alpha
        self.split = split
        # Initialize the low-rank matrices A and B
        assert self.out_features % split == 0
        self.As = nn.ParameterList([
            nn.Parameter(torch.randn(self.in_features, rank, device = module.weight.device, dtype = module.weight.dtype) * 1 / math.sqrt(rank), requires_grad=True)
            for _ in range(split)
        ])
        self.Bs = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, self.out_features // split, device = module.weight.device, dtype=module.weight.dtype), requires_grad=True)
            for _ in range(split)
        ])

    def forward(self, x, *args, **kwargs):
        # print('LoRALayer forward', x.size(), self.A.size(), self.B.size(), 'rank:', self.rank, 'alpha:', self.alpha)
        # Apply LoRA: x += x @ A @ B, where @ denotes matrix multiplication
        xs = [x@A@B for A, B in zip(self.As, self.Bs)]
        lora_adaptation = torch.cat(xs, dim = -1) * self.alpha
        return self.linear(x) + lora_adaptation

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}, split={self.split})")

class LoRaMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, method, rank, profile: bool = False, save_dir = None, tag = None):
        # print('LowRankApproximationFunction forward', input.size(), weight.size(), rank_approx)
        ctx.save_for_backward(input, weight)
        ctx.method = method
        ctx.rank = rank
        ctx.profile = profile
        ctx.save_dir = save_dir
        ctx.tag = tag
        output = input @ weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors # input: B x SL x HS, weight: HS x HS, S: SL x rank_approx
        
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            # For the gradient w.r.t. the input, if needed
            grad_input = grad_output @ weight.t()

        if ctx.needs_input_grad[1]:
            # grad_output: B x SL x HS, input: B x SL x HS, Weight HS x HS
            if ctx.method in APPROX_MAP:
                func = APPROX_MAP[ctx.method]
            else: # Default
                func = lambda x, y, _: x @ y
            
            # if input.is_contiguous():
            #     input_ = input.view(..., input.size(-1)).t()
            # else: 
            input_ = input.reshape(-1, input.size(-1)).t()
            # if grad_input.is_contiguous():
            #     grad_output_ = grad_output.view(-1, grad_output.size(-1))
            # else:
            grad_output_ = grad_output.reshape(-1, grad_output.size(-1))
            if len(inspect.signature(func).parameters) == 2:
                grad_weight = func(input_@grad_output_, ctx.rank)
            elif len(inspect.signature(func).parameters) == 3:
                grad_weight = func(input_, grad_output_, ctx.rank)
        if ctx.profile:
            matrices = [('input', input), ('grad_weight', grad_weight), ('grad_output', grad_output), ('weight', weight)]
            matrix_analyzer(matrices, save_dir = ctx.save_dir, tag = ctx.tag)
        
        return grad_input, grad_weight, None, None, None, None, None

class LoRaLinear(nn.Module):
    def __init__(self, 
                 module: Union[nn.Linear, transformers.pytorch_utils.Conv1D], 
                 method = None, 
                 rank = None, 
                 optim = None):
        super(LoRaLinear, self).__init__()
        if isinstance(module, nn.Linear):
            self.in_features = module.in_features
            self.out_features = module.out_features
            self.weight = torch.nn.Parameter(module.weight.data.t())
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            self.in_features = module.weight.size(0)
            self.out_features = module.weight.size(1)
            self.weight = module.weight
        self.weight.requires_grad = True
        self.bias = module.bias
        self.method = method
        self.rank = rank
    # self.c_attn(hidden_states, profile, self.global_state['profile_dir'], f'L{self.layer_idx}_QKV')
    def forward(self, input, profile = False, save_dir = None, tag = None):
        ret = LoRaMatmulFunction.apply(input, self.weight, self.method, self.rank, profile, save_dir, tag)
        if self.bias is not None:
            ret += self.bias
        return ret

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, method={self.method}, rank={self.rank})")

def prepare_lora_grad(model, method, rank, modules = ['c_attn']):
    for name, param in model.named_parameters():
        param.requires_grad = False
    def recursive_replace(module, tag = []):
        for name, submodule in module.named_children():
            if any([(name.find(m) != -1) or m in tag for m in modules]) and\
            isinstance(submodule, nn.Linear) or isinstance(submodule, transformers.pytorch_utils.Conv1D):
                setattr(module, name, LoRALayer(submodule, method, rank))
            else: 
                recursive_replace(submodule, tag + [name])
    recursive_replace(model)
    return model

def prepare_lora(model, rank, alpha = 8, modules = ['c_attn']):
    for name, param in model.named_parameters():
        param.requires_grad = False
    def recursive_replace(module, tag = []):
        for name, submodule in module.named_children():
            if any([(name.find(m) != -1) or m in tag for m in modules]) and\
            isinstance(submodule, nn.Linear) or isinstance(submodule, transformers.pytorch_utils.Conv1D):
                n_split = 3 if name == 'c_attn' else 1
                setattr(module, name, LoRALayer(submodule, rank, alpha, n_split))
            else: 
                recursive_replace(submodule, tag + [name])
    recursive_replace(model)
    return model

class SelfAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, profile = False, save_dir = None, tag = None):
        # query: [batch_size, num_heads, seq_len, head_dim]
        # key: [batch_size, num_heads, PSL + seq_len, head_dim]
        # value: [batch_size, num_heads, PSL + seq_len, head_dim]
        # mask: [batch_size, num_heads, seq_len, seq_len]
        QK = torch.matmul(query, key.transpose(-2, -1)) # [batch_size, num_heads, seq_len, PSL + seq_len]
        if mask is not None:
            QK += mask
        QK = QK / (key.size(-1) ** 0.5)
        A = torch.nn.functional.softmax(QK, dim=-1) # [batch_size, num_heads, seq_len, PSL + seq_len]
        output = torch.matmul(A, value)
        ctx.save_for_backward(query, key, value, A)
        ctx.profile = profile
        ctx.save_dir = save_dir
        ctx.tag = tag
        return output, A

    @staticmethod
    def backward(ctx, grad_output, grad_A = None):
        # grad_output: [batch_size, num_heads, seq_len, head_dim]
        query, key, value, A = ctx.saved_tensors
        grad_V = A.transpose(-2, -1) @ grad_output # [batch_size, num_heads, seq_len, head_dim]
        grad_A = grad_output @ value.transpose(-2, -1) # [batch_size, num_heads, seq_len, PSL + seq_len]
        # print('grad_A', grad_A.size(), 'A', A.size(), 'key', key.size())
        grad_AA = grad_A * A
        grad_QK = ((grad_AA) - (grad_AA).sum(-1).unsqueeze(-1) * A) / (key.size(-1) ** 0.5) # [batch_size, num_heads, seq_len, PSL + seq_len]
        grad_Q = grad_QK @ key # [batch_size, num_heads, seq_len, head_dim]
        grad_K = grad_QK.transpose(-2, -1) @ query # [batch_size, num_heads, PSL + seq_len, head_dim]
        # if ctx.profile:
        #     matrices = [('grad_Q', grad_Q), ('grad_K', grad_K), ('grad_V', grad_V), ('grad_A', grad_A), ('A', A)]
        #     matrix_analyzer(matrices, save_dir = ctx.save_dir, tag = ctx.tag)

        return grad_Q, grad_K, grad_V, None, None, None, None

class SelfAttention(torch.nn.Module):
    def forward(self, query, key, value, mask):
        return SelfAttentionFunction.apply(query, key, value, mask)

class SelfAttentionGroundTruth(torch.nn.Module):
    def forward(self, query, key, value, mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if mask is not None:
            qk += mask
        return torch.softmax(qk / (key.size(-1) ** 0.5), dim=-1) @ value

def test_self_attn():
    import time
    bs, num_heads, seq_len, head_dim = 2, 3, 4, 5
    PSL = 6
    query = torch.randn(bs, num_heads, seq_len, head_dim).requires_grad_(True)
    key = torch.randn(bs, num_heads, PSL + seq_len, head_dim).requires_grad_(True)
    value = torch.randn(bs, num_heads, PSL + seq_len, head_dim).requires_grad_(True)
    mask = torch.randn(bs, num_heads, seq_len, PSL + seq_len)
    self_attn = SelfAttention()
    
    query_gt = query.detach().clone().requires_grad_(True)
    key_gt = key.detach().clone().requires_grad_(True)
    value_gt = value.detach().clone().requires_grad_(True)
    mask_gt = mask.detach().clone()

    self_attn_gt = SelfAttentionGroundTruth()
    out = self_attn(query, key, value, mask)
    l = out.sum()
    out_gt = self_attn_gt(query_gt, key_gt, value_gt, mask_gt)
    assert torch.allclose(out, out_gt)
    
    l_gt = out_gt.sum()
    l.backward()
    l_gt.backward()
    
    assert torch.allclose(value.grad, value_gt.grad)
    assert torch.allclose(query.grad, query_gt.grad)
    assert torch.allclose(key.grad, key_gt.grad)
    
    start = time.time()
    for i in range(100):
        out = self_attn(query, key, value, mask)
        l = out.sum()
        l.backward()
    print('time', time.time() - start)
    start = time.time()
    
    for i in range(100):
        out_gt = self_attn_gt(query_gt, key_gt, value_gt, mask_gt)
        l_gt = out_gt.sum()
        l_gt.backward()
    print('time', time.time() - start)

from offload_modules import LoRaOffloadLinear

class LoRaMHA(nn.Module):
    def __init__(self,
                attn: transformers.models.gpt2.modeling_gpt2.GPT2Attention,
                method: str = None,
                rank: int = None,
                modules: list = ['attn.qkv'],
                offload: str = None,
                lr: float = None,
                delta_compress_method: str = None,
                delta_compress_rank: str = None,
                global_state: dict = {},
                tag: str = None):
        super().__init__()
        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.split_size = attn.split_size
        self.scale_attn_weights = attn.scale_attn_weights
        self.is_cross_attention = attn.is_cross_attention
        self.scale_attn_by_inverse_layer_idx = attn.scale_attn_by_inverse_layer_idx
        self.layer_idx = attn.layer_idx
        self.reorder_and_upcast_attn = attn.reorder_and_upcast_attn
        if 'attn.qkv' in modules:
            if method == 'lora': 
                self.c_attn = LoRALayer(attn.c_attn, rank)
            elif offload is None:
                self.c_attn = LoRaLinear(attn.c_attn, method, rank)
            else: 
                self.c_attn = LoRaOffloadLinear(attn.c_attn, method, rank, lr, offload, delta_compress_method, delta_compress_rank, global_state, tag)
        else:
            self.c_attn = LoRaLinear(attn.c_attn, None, None)
            self.c_attn.weight.requires_grad = False
        if 'attn.o' in modules:
            if method == 'lora': 
                self.c_proj = LoRALayer(attn.c_proj, rank)
            elif offload is None:
                self.c_proj = LoRaLinear(attn.c_proj, method, rank)
            else:
                self.c_proj = LoRaOffloadLinear(attn.c_proj, method, rank, lr, offload, delta_compress_method, delta_compress_rank, global_state, tag)
        else: 
            self.c_proj = LoRaLinear(attn.c_proj, None, None)
            self.c_proj.weight.requires_grad = False
        
        self.attn_dropout = attn.attn_dropout
        self.resid_dropout = attn.resid_dropout
        self.global_state = global_state
        self.tag = tag

        # max_positions = config.max_position_embeddings
        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
        #         1, 1, max_positions, max_positions
        #     ),
        #     persistent=False,
        # )
        # self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        assert encoder_hidden_states is None
        profile = 'profile' in self.global_state and self.global_state['profile']
        query, key, value = self.c_attn(hidden_states, profile, self.global_state['profile_dir'], f'L{self.layer_idx}_QKV').split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim) # [bs, num_heads, seq_len, head_dim]
        key = self._split_heads(key, self.num_heads, self.head_dim) # [bs, num_heads, PSL + seq_len, head_dim]
        value = self._split_heads(value, self.num_heads, self.head_dim) # [bs, num_heads, PSL + seq_len, head_dim]

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            attn_output, attn_weights = SelfAttentionFunction.apply(query, key, value, attention_mask, profile, self.global_state['profile_dir'], f'L{self.layer_idx}_MHA')

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
    def __repr__(self):
        return f'MultiHeadSelfAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}, c_attn={self.c_attn}, c_proj={self.c_proj}, attn_dropout={self.attn_dropout}, resid_dropout={self.resid_dropout} )'

def replace_attn_layer(model, **kwargs):
    def replace(module, tag = ''):
        for name, child in module.named_children():
            if isinstance(child, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
                attn = LoRaMHA(child, **kwargs, tag= tag + name + '.')
                setattr(module, name, attn)
            else:
                replace(child, tag + name + '.')
        return module   
    return replace(model)

from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertIntermediate
from transformers.models.roberta.modeling_roberta import RobertaSelfOutput, RobertaOutput, RobertaIntermediate


def replace_linear_layer(model, 
                        method, 
                        rank, 
                        modules, 
                        offload,
                        lr,
                        delta_compress_method,
                        delta_compress_rank, 
                        global_state,
                        memory_cap,
                        optim_dtype):
    global_state['offload_modules'] = []
    def replace(module, tag = '', memory_cap = memory_cap):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) or isinstance(child, transformers.pytorch_utils.Conv1D):
                if ('attn.qkv' in modules and name in ('q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'c_attn')) or\
                    ('attn.o' in modules and name in ('o_proj')) or \
                    ('attn.o' in modules and name == 'dense' and isinstance(module, BertSelfOutput)) or\
                    ('attn.o' in modules and name == 'dense' and isinstance(module, RobertaSelfOutput)) or\
                    ('mlp.gate' in modules and name in ('gate_proj')) or\
                    ('mlp.up' in modules and name in ('up_proj', 'c_fc')) or\
                    ('mlp.up' in modules and name == 'dense' and isinstance(module, BertIntermediate)) or\
                    ('mlp.up' in modules and name == 'dense' and isinstance(module, RobertaIntermediate)) or\
                    ('mlp.down' in modules and name in ('down_proj', 'c_proj')) or\
                    ('mlp.down' in modules and name == 'dense' and isinstance(module, BertOutput))or\
                    ('mlp.down' in modules and name == 'dense' and isinstance(module, RobertaOutput)):
                    if method == 'lora': 
                        new_module = LoRALayer(child, rank)
                    elif offload is None:
                        new_module = LoRaLinear(child, method, rank)
                    else:
                        mem_indemand = child.weight.numel() * child.weight.element_size() * 2 # adam
                        if child.bias is not None:
                            mem_indemand += child.bias.numel() * child.bias.element_size() * 2
                        if mem_indemand < memory_cap:
                            # no offloading
                            new_module = LoRaOffloadLinear(child, None, None, lr = lr, offloading_device=child.weight.device.__str__())
                            memory_cap -= mem_indemand
                        else: 
                            new_module = LoRaOffloadLinear(child, method, rank, lr, offload, delta_compress_method, delta_compress_rank, global_state, tag + name, optim_dtype)
                    setattr(module, name, new_module)
            else:
                memory_cap = replace(child, tag + name + '.', memory_cap)
        return memory_cap
    print('memory cap for optim state:', memory_cap / 1e9, 'GB')
    memory_cap = replace(model)
    print('remaining memory :', memory_cap / 1e9, 'GB')
    return model

class LoRaAdam:
    def __init__(self, params, lr = 1e-3, beta = 0.9, gamma = 0.999, eps = 1e-4, approx_method: str = None, approx_rank = None):
        # defaults = dict(lr=lr, beta=beta, gamma=gamma, eps=eps, approx_method=approx_method, approx_rank=approx_rank)
        # super(LoRaAdam, self).__init__(params, defaults)
        self.params = [p for p in params if p.requires_grad]
        # on the same device as p
        self.us = [torch.zeros_like(p) for p in self.params]
        self.vs = [torch.zeros_like(p) for p in self.params]
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.t = 0
        self.approx_method = approx_method
        if approx_method == 'lora_approx':
            self.approx_impl = lora_approx
        elif approx_method == 'lora_svd':
            self.approx_impl = lora_svd
        elif approx_method == 'lora_power':
            self.approx_impl = lora_power
        elif approx_method == 'lora_gaussian':
            self.approx_impl = lora_gaussian
        elif approx_method == 'topk':
            self.approx_impl = topk
        else: # no approximation
            self.approx_impl = lambda x, _: x
        self.approx = lambda x, rank: self.apporx_impl(x, rank) if x.size() == 2 else x
        self.approx_rank = approx_rank
    
    def state_dict(self):
        """Returns the state of the optimizer as a dictionary."""
        state = super().state_dict()
        state['us'] = self.us
        state['vs'] = self.vs
        state['t'] = self.t
        return state

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        super().load_state_dict(state_dict)
        self.us = state_dict['us']
        self.vs = state_dict['vs']
        self.t = state_dict['t']

    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, u, v in zip(self.params, self.us, self.vs):
                u[:] = self.beta * u + (1-self.beta) * param.grad
                v[:] = self.gamma * v + (1-self.gamma) * param.grad**2
                u_hat = u / (1 - self.beta**self.t)
                v_hat = v / (1 - self.gamma**self.t)
                delta = u_hat / (torch.sqrt(v_hat) + self.eps)
                # print('delta:', delta.min().item(), delta.max().item(), 'lr', self.lr)
                update = self.lr * delta
                param -= self.approx(update, self.approx_rank)

    def zero_grad(self):
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
                
    def __repr__(self):
        return f'LoRaAdam(n_param: {len(self.params)}, lr={self.lr}, beta={self.beta}, gamma={self.gamma}, eps={self.eps}, approx_method={self.approx_method}, approx_rank={self.approx_rank})'

import tqdm

def test_optimizer():
    W = torch.randn(5, 5).cuda()
    W_param = torch.nn.Parameter(torch.randn(5, 5).cuda(), requires_grad=True)
    optim = LoRaAdam([W_param], lr = 1e-2)
    tqdm_struct = tqdm.tqdm(range(1000))
    acc_loss = None
    for i in tqdm_struct:
        X = torch.randn(10, 5).cuda()
        y = X @ W_param
        with torch.no_grad():
            y_gt = X @ W
        loss = (y - y_gt).pow(2).mean()
        acc_loss = loss.item() if acc_loss is None else 0.9 * acc_loss + 0.1 * loss.item()
        tqdm_struct.set_description(f'loss: {acc_loss}')
        optim.zero_grad()
        loss.backward()
        optim.step()

if __name__ == '__main__':
    test_optimizer()
    # test_self_attn()
    print('test passed')