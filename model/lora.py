from collections import OrderedDict
from typing import Dict
import typing
import torch


def get_filter_keys_and_merge_coefficients(layer_filter):
    if layer_filter:
        layers = []
        layer_coefficients = {}
        for layer in layer_filter.split(' '):
            if '*' in layer:
                coefficient, _, layer = layer.partition('*')
                coefficient = float(coefficient)
            else:
                coefficient = 1
            if layer.isdecimal():
                layers.append(int(layer))
                layer_coefficients[int(layer)] = coefficient
            elif '-' in layer:
                start, _, end = layer.partition('-')
                start, end = int(start), int(end)
                layers.extend(range(start, end+1))
                for l in range(start, end+1):
                    layer_coefficients[l] = coefficient
            else:
                raise NotImplementedError(
                    "layer_filter Not implemented:", layer_filter)
        layers = sorted(set(layers))
        layer_prefixes = tuple(f"blocks.{l}." for l in layers)

        def filter_keys(keys):
            new_keys = []
            for key in keys:
                # Skip weights that are started by 'blocks.' and not in allowed range
                if key.startswith("blocks.") and not key.startswith(layer_prefixes):
                    continue
                new_keys.append(key)
            return new_keys

        def merge_coefficients(key):
            if key.startswith('blocks.') and int(key.split('.')[1]) in layer_coefficients:
                return layer_coefficients[int(key.split('.')[1])]
            else:
                return 1
    else:
        def filter_keys(keys):
            return keys

        def merge_coefficients(key):
            return 1
    return filter_keys, merge_coefficients


def lora_merge(base_model, lora, lora_alpha, device="cuda", layer_filter=None,):
    print(f"Loading LoRA: {lora}")
    print(f"LoRA alpha={lora_alpha}, layer_filter={layer_filter}")
    filter_keys, merge_coef = get_filter_keys_and_merge_coefficients(
        layer_filter)
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')
    # pdb.set_trace() #DEBUG
    for k in filter_keys(w_lora.keys()):  # 处理time_mixing之类的融合
        w[k] = w_lora[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge LoRA weights
    keys = list(w.keys())
    for k in keys:
        if k.endswith('.weight'):
            prefix = k[:-len('.weight')]
            lora_A = prefix + '.lora_A'
            lora_B = prefix + '.lora_B'
            if lora_A in keys:
                assert lora_B in keys
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
                w[k] += w[lora_B] @ w[lora_A] * \
                    (lora_alpha / lora_r) * merge_coef(k)
                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[lora_A]
                del w[lora_B]
                continue

        if 'lora' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]
    return output_w
