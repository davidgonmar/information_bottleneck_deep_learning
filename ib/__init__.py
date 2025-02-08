import torch
from typing import Sequence, Callable
from collections import Counter
import numpy as np
import math


@torch.no_grad()
def get_mutual_info(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    layers: Sequence[torch.nn.Module],
    p_x: Callable[[torch.Tensor], torch.Tensor],
    p_y: Callable[[torch.Tensor], torch.Tensor],
    get_bin_config: Callable[[str], tuple] = lambda x: (-1, 1, 30),
):
    layers = tuple(layers)

    current_labels = None
    current_xs = None
    hidden_layer_stats = {}

    device = next(model.parameters()).device

    for layer in model.modules():
        if isinstance(layer, layers):
            hidden_layer_stats[layer] = {
                "hist_t": Counter(),
                "hist_tandy": Counter(),
                "hist_tandx": Counter(),
            }

    def append_stats(layer, input, output):
        nonlocal current_labels
        nonlocal current_xs
        if isinstance(layer, layers):
            # output has shape (batch_size, out_features)
            # each el of output (of shape out_features)
            # digitized into 30 bins
            digitized = np.digitize(
                output.cpu().numpy(), np.linspace(*get_bin_config(layer))
            )
            li = digitized.tolist()
            for ll in li:
                hidden_layer_stats[layer]["hist_t"][tuple(ll)] += 1

            # now, for each label, we increment the corresponding entry
            for i in range(output.size(0)):
                hidden_layer_stats[layer]["hist_tandy"][
                    tuple(li[i]) + (current_labels[i].item(),)
                ] += 1

            # now, for each x, we increment the corresponding entry
            for i in range(output.size(0)):
                current_x = (
                    current_xs[i].cpu().numpy().tolist()
                )  # binary repr of x (between 0 and 1024)
                hidden_layer_stats[layer]["hist_tandx"][
                    tuple(li[i]) + (tuple(current_x),)
                ] += 1

    for layer in model:
        # clear all hooks
        layer._forward_hooks.clear()
        layer.register_forward_hook(append_stats)

    with torch.no_grad():
        for images, labels in test_loader:
            current_labels = labels
            current_xs = images
            images, labels = images.to(device), labels.to(device)
            model(images)

    # count total observations
    total_observations = sum(hidden_layer_stats[model[1]]["hist_t"].values())

    for layer, info in hidden_layer_stats.items():
        info["p_t"] = info["hist_t"].copy()
        for k in info["p_t"]:
            info["p_t"][k] /= total_observations

        info["p_tandy"] = info["hist_tandy"].copy()
        for k in info["p_tandy"]:
            info["p_tandy"][k] /= total_observations

        info["p_tandx"] = info["hist_tandx"].copy()
        for k in info["p_tandx"]:
            info["p_tandx"][k] /= total_observations

    mis_y = dict()
    mis_x = dict()
    for layer, info in hidden_layer_stats.items():
        mi = 0
        for k, v in info["p_tandy"].items():
            t = k[:-1]
            py = p_y(torch.tensor(k[-1]).to(device))
            mi += v * math.log(v / (info["p_t"][t] * py))
        mis_y[layer] = mi

        mi = 0
        for k, v in info["p_tandx"].items():
            t = k[:-1]
            px = p_x(torch.tensor(k[-1]).to(device))
            mi += v * math.log(v / (info["p_t"][t] * px))
        mis_x[layer] = mi

    for layer in model:
        layer._forward_hooks.clear()

    return mis_y, mis_x
