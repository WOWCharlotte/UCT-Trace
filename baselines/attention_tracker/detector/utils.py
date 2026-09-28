import torch 
import numpy as np

def process_attn(attention, rng, attn_func):
    heatmap = torch.zeros((len(attention), attention[0].shape[1]), dtype=torch.float32)
    for i, attn_layer in enumerate(attention):
        attn_layer = attn_layer.to(torch.float32)

        if "sum" in attn_func:
            last_token_attn_to_inst = attn_layer[0, :, -1, rng[0][0]:rng[0][1]].sum(dim=1)
            attn = last_token_attn_to_inst
        
        elif "max" in attn_func:
            last_token_attn_to_inst = attn_layer[0, :, -1, rng[0][0]:rng[0][1]].max(dim=1).values
            attn = last_token_attn_to_inst

        else: raise NotImplementedError
            
        last_token_attn_to_inst_sum = attn_layer[0, :, -1, rng[0][0]:rng[0][1]].sum(dim=1)
        last_token_attn_to_data_sum = attn_layer[0, :, -1, rng[1][0]:rng[1][1]].sum(dim=1)

        if "normalize" in attn_func:
            epsilon = 1e-8
            heatmap[i, :] = attn / (last_token_attn_to_inst_sum + last_token_attn_to_data_sum + epsilon)
        else:
            heatmap[i, :] = attn

    heatmap = torch.nan_to_num(heatmap, nan=0.0)

    return heatmap


def calc_attn_score(heatmap, heads):
    score = torch.stack([heatmap[l, h] for l, h in heads]).mean()
    return float(score.item())
