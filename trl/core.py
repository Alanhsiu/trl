# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import random
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

from .import_utils import is_npu_available, is_xpu_available


try:
    from collections.abc import Mapping
except ImportError:
    from collections.abc import Mapping


WANDB_PADDING = -1


def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits


def flatten_dict(nested: Dict, sep: str = "/") -> Dict:
    """Flatten dictionary and concatenate nested keys with separator."""

    def recurse(nest: Dict, prefix: str, into: Dict) -> None:
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, Mapping):
                recurse(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    recurse(nested, "", flat)
    return flat


def convert_to_scalar(stats: Dict) -> Dict:
    """
    Converts the stats from a flattened dict to single scalar dicts
    """
    tensorboard_stats = {}
    for k, v in stats.items():
        # for tensorboard compatibility - arrays and tensors are ignored with tensorboard
        # therefore we convert single element tensors to scalars
        if (isinstance(v, torch.Tensor) or isinstance(v, np.ndarray)) and (
            len(v.shape) == 0 or (len(v.shape) == 1 and v.shape[0] == 1)
        ):
            v = v.item()
        tensorboard_stats[k] = v
    return tensorboard_stats


def stack_dicts(stats_dicts: List[Dict]) -> Dict:
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
    return results


def add_suffix(input_dict: Dict, suffix: str) -> Dict:
    """Add suffix to dict keys."""
    return {k + suffix: v for k, v in input_dict.items()}


def pad_to_size(tensor: torch.Tensor, size: int, dim: int = 1, padding: int = 50256) -> torch.Tensor:
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size == size:
        return tensor
    else:
        return torch.nn.functional.pad(tensor, (0, size - t_size), "constant", padding)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)

    if not gather:
        return logp
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def whiten(values: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[bool] = None) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x: torch.Tensor, tensor_min: float, tensor_max: float) -> torch.Tensor:
    """
    Tensor extension to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts: List[Dict]) -> Dict:
    """Average values of a list of dicts with torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict


def stats_to_np(stats_dict: Dict) -> Dict:
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu()
            if new_dict[k].dtype == torch.bfloat16:
                new_dict[k] = new_dict[k].float()
            new_dict[k] = new_dict[k].numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict

import torch
import torch.nn.functional as F
from torch import nn
from typing import List

# def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
#     """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
#     # ... your implementation here ...
#     return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    if top_k > 0:
        top_k = min(max(top_k, 1), logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def old_respond_to_batch(
    model: nn.Module, queries: List[torch.LongTensor], txt_len: int = 20, top_k: int = 0, top_p: float = 1.0
) -> torch.LongTensor:
    """Sample text from language model."""
    input_ids = queries
    count = 0
    response = []
    for _i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        response.append(next_token)
        if (next_token == 2).any():  # EOS token
            break
        count+=1
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    # return input_ids[:, -count:]
    response = torch.stack(response).transpose(0, 1)  # convert list of tensors to a single tensor
    return response

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def respond_to_batch(
    model: nn.Module, queries: List[torch.LongTensor], txt_len: int = 100, top_k: int = 0, top_p: float = 1.0, tokenizer=None
) -> torch.LongTensor:
    
    input_ids = queries
    count = 0
    min_steps = 5
    response = []
    
    for _i in range(txt_len):
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        if count == 0:
            print(f"i: {_i}, next_token_logits shape: {next_token_logits.shape}, size: {next_token_logits.size()}, size(-1): {next_token_logits.size(-1)}")
        
        if next_token_logits.size(-1) < 50264:
            raise RuntimeError(f"Logits dim error: {next_token_logits.size()}")
        
        if count < min_steps:
            next_token_logits[:, 2] = -float("Inf")
        next_token_logits[:, 4:50264] = -float("Inf")
        next_token_logits[:, 51289:58457] = -float("Inf")
        
        # Apply top-k and top-p filtering
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        
        # Checks for invalid values (infinity, NaN, negative values) in the logits and replaces them with a large negative number to avoid errors.
        if torch.isinf(next_token_logits).any() or torch.isnan(next_token_logits).any() or (next_token_logits < 0).any():
            print(f"Invalid values found in next_token_logits at step {_i}")
            # Replace invalid values with a large negative number (e.g., -1e10) to avoid errors
            next_token_logits = torch.where(
                torch.isinf(next_token_logits) | torch.isnan(next_token_logits) | (next_token_logits < 0),
                torch.tensor(-1e10).to(next_token_logits.device),
                next_token_logits
            )
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        if (next_token >= next_token_logits.size(-1)).any() or (next_token < 0).any():
            raise ValueError(f"Index out of range: {next_token}")

        response.append(next_token)
        print("next_token: ", next_token, " dimension: ", next_token.dim())
        print("response: ", response)
        
        decoded_next_token = tokenizer.decode(next_token)

        if next_token == 2:
            print("input_ids: ", input_ids)
            print(f"count: {count}, input_ids[:, -count:]: {input_ids[:, -count:]}, input_ids shape: {input_ids.shape}, size: {input_ids.size()}, size(1): {input_ids.size(1)}")
            break
        if input_ids.shape[1] >= 1024:
            print("input_ids reached max length")
            print("input_ids: ", input_ids)
            print(f"count: {count}, input_ids[:, -count:]: {input_ids[:, -count:]}, input_ids shape: {input_ids.shape}, size: {input_ids.size()}, size(1): {input_ids.size(1)}")
            break
        
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        count += 1

    response = torch.stack(response).transpose(0, 1)  # convert list of tensors to a single tensor
    return response


# def respond_to_batch(
#     model: nn.Module, queries: List[torch.LongTensor], txt_len: int = 100, top_k: int = 0, top_p: float = 1.0, tokenizer=None
# ) -> torch.LongTensor:
    
#     input_ids = queries
#     count = 0
#     min_steps = 5
#     response = []
    
#     for _i in range(txt_len):
#         outputs = model(input_ids)
#         next_token_logits = outputs[0][:, -1, :]
#         if count == 0:
#             print(f"i: {_i}, next_token_logits shape: {next_token_logits.shape}, size: {next_token_logits.size()}, size(-1): {next_token_logits.size(-1)}")
        
#         if next_token_logits.size(-1) < 50264:
#             raise RuntimeError(f"Logits dim error: {next_token_logits.size()}")
        
#         if count < min_steps:
#             next_token_logits[:, 2] = -float("Inf")
#         next_token_logits[:, 4:50264] = -float("Inf")
#         next_token_logits[:, 51289:58457] = -float("Inf")
        
#         # Apply top-k and top-p filtering
#         next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        
#         # Checks for invalid values (infinity, NaN, negative values) in the logits and replaces them with a large negative number to avoid errors.
#         if torch.isinf(next_token_logits).any() or torch.isnan(next_token_logits).any() or (next_token_logits < 0).any():
#             print(f"Invalid values found in next_token_logits at step {_i}")
#             # Replace invalid values with a large negative number (e.g., -1e10) to avoid errors
#             next_token_logits = torch.where(
#                 torch.isinf(next_token_logits) | torch.isnan(next_token_logits) | (next_token_logits < 0),
#                 torch.tensor(-1e10).to(next_token_logits.device),
#                 next_token_logits
#             )
        
#         probs = F.softmax(next_token_logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

#         if (next_token >= next_token_logits.size(-1)).any() or (next_token < 0).any():
#             raise ValueError(f"Index out of range: {next_token}")

#         # input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
#         response.append(next_token)
#         print("next_token: ", next_token, " dimension: ", next_token.dim())
#         print("response: ", response)
        
#         decoded_next_token = tokenizer.decode(next_token)

#         # if input_ids[0, -1] == 2:
#         if next_token == 2:
#             print("input_ids: ", input_ids)
#             print(f"count: {count}, input_ids[:, -count:]: {input_ids[:, -count:]}, input_ids shape: {input_ids.shape}, size: {input_ids.size()}, size(1): {input_ids.size(1)}")
#             break
#         if input_ids.shape[1] >= 1024:
#             print("input_ids reached max length")
#             print("input_ids: ", input_ids)
#             print(f"count: {count}, input_ids[:, -count:]: {input_ids[:, -count:]}, input_ids shape: {input_ids.shape}, size: {input_ids.size()}, size(1): {input_ids.size(1)}")
#             break
        
#         count += 1
        
#         print("torch.stack(response)", torch.stack(response))

#     # return input_ids[:, -count:]
#     # return torch.stack(response)
#     return response

# def respond_to_batch(
#     model: nn.Module, queries: List[torch.LongTensor], txt_len: int = 100, top_k: int = 0, top_p: float = 1.0, tokenizer=None
# ) -> torch.LongTensor:
    
#     input_ids = queries
#     count = 0
#     min_steps = 5
    
#     for _i in range(txt_len):
#         outputs = model(input_ids)
#         next_token_logits = outputs[0][:, -1, :]
#         if count == 0:
#             print(f"i: {_i}, next_token_logits shape: {next_token_logits.shape}, size: {next_token_logits.size()}, size(-1): {next_token_logits.size(-1)}")
        
#         if next_token_logits.size(-1) < 50264:
#             raise RuntimeError(f"Logits dim error: {next_token_logits.size()}")
        
#         # if count < min_steps:
#         #     next_token_logits[:, 2] = -float("Inf")
#         next_token_logits[:, 4:50264] = -float("Inf")
#         next_token_logits[:, 51289:58457] = -float("Inf")
#         next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

#         if torch.isinf(next_token_logits).any() or torch.isnan(next_token_logits).any() or (next_token_logits < 0).any():
#             print(f"Invalid values found in next_token_logits at step {_i}")
#             # Replace invalid values with a large negative number (e.g., -1e10) to avoid errors
#             next_token_logits = torch.where(
#                 torch.isinf(next_token_logits) | torch.isnan(next_token_logits) | (next_token_logits < 0),
#                 torch.tensor(-1e10).to(next_token_logits.device),
#                 next_token_logits
#             )

#         probs = F.softmax(next_token_logits, dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

#         if (next_token >= next_token_logits.size(-1)).any() or (next_token < 0).any():
#             raise ValueError(f"Index out of range: {next_token}")

#         # if (next_token == 2).any():  # EOS token
#         #     break
        
#         input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        
#         decoded_next_token = tokenizer.decode(next_token)

#         if input_ids[0, -1] == 2:
#             print(f"count: {count}, input_ids[:, -count:]: {input_ids[:, -count:]}, input_ids shape: {input_ids.shape}, size: {input_ids.size()}, size(1): {input_ids.size(1)}")
#             break
#         if input_ids.shape[1] >= 1024:
#             print("input_ids reached max length")
#             print(f"count: {count}, input_ids[:, -count:]: {input_ids[:, -count:]}, input_ids shape: {input_ids.shape}, size: {input_ids.size()}, size(1): {input_ids.size(1)}")
#             break
        
#         count += 1

#     # print("input_ids: ", input_ids)
#     # return input_ids[:, -txt_len:]
#     return input_ids[:, -count:]

# def respond_to_batch(
#     model: nn.Module, queries: List[torch.LongTensor], txt_len: int = 100, top_k: int = 0, top_p: float = 1.0, tokenizer=None
# ) -> torch.LongTensor:
#     """Sample text from language model."""
#     input_ids = queries
#     print("queries: ", queries)
#     for _i in range(txt_len):
#         # Get Logits
#         outputs = model(input_ids)
#         print("i: ", _i, "outputs shape: ", outputs[0].shape, "outputs: ", outputs)
#         print("outputs[0]: ", outputs[0])
#         print("outputs[0][:, -1, :]: ", outputs[0][:, -1, :])
#         next_token_logits = outputs[0][:, -1, :]
#         next_token_logits[:, 4:50264] = -float("Inf")
#         next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        
#         print("i: ", _i, "next_token_logits shape: ", next_token_logits.shape, "next_token_logits: ", next_token_logits)
        
#         # Sample
#         probs = F.softmax(next_token_logits, dim=-1)
#         print("i: ", _i, "probs shape: ", probs.shape, "probs: ", probs)
#         next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
#         print("i: ", _i, "next_token shape: ", next_token.shape, "next_token: ", next_token)
        
#         decoded_next_token = tokenizer.decode(next_token)
#         print("i: ", _i, "decoded_next_token: ", decoded_next_token)
#         if (next_token == 2).any():
#             break
#         input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
#         print("i: ", _i, "input_ids: ", input_ids)
        

#     return input_ids[:, -txt_len:]


def set_seed(seed: int) -> None:
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed_all(seed)
    elif is_npu_available():
        torch.npu.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)


class LengthSampler:
    """
    Samples a length
    """

    def __init__(self, min_value: int, max_value: int):
        self.values = list(range(min_value, max_value))

    def __call__(self) -> int:
        return np.random.choice(self.values)


class PPODecorators:
    optimize_device_cache = False

    @classmethod
    @contextmanager
    def empty_device_cache(cls):
        yield
        if cls.optimize_device_cache:
            if is_xpu_available():
                gc.collect()
                torch.xpu.empty_cache()
                gc.collect()
            elif is_npu_available():
                gc.collect()
                torch.npu.empty_cache()
                gc.collect()
            elif torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List[torch.Generator], torch.Generator]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
) -> torch.Tensor:
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                warnings.warn(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents
