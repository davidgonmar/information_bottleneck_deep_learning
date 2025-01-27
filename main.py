import torch


@torch.no_grad()
def get_mutual_info(model):
    # each layer out will be digitized using 30 bins between -1 and 1
    from collections import Counter
    import numpy as np

    current_labels = None
    current_xs = None
    hidden_layer_stats = {}
    import math

    for layer in model:
        if isinstance(layer, LinearAndActivation):
            hidden_layer_stats[layer] = {
                "hist_t": Counter(),
                "hist_tandy": Counter(),
                "hist_tandx": Counter(),
            }

    def append_stats(layer, input, output):
        nonlocal current_labels
        nonlocal current_xs
        if isinstance(layer, LinearAndActivation):
            # output has shape (batch_size, out_features)
            # each el of output (of shape out_features)
            # digitized into 30 bins
            digitized = np.digitize(output.cpu().numpy(), np.linspace(-1, 1, 30))
            li = digitized.tolist()
            for ll in li:
                hidden_layer_stats[layer]["hist_t"][tuple(ll)] += 1

            # now, for each label, we increment the corresponding entry
            for i in range(output.size(0)):
                hidden_layer_stats[layer]["hist_tandy"][
                    tuple(li[i]) + (current_labels[i].item(),)
                ] += 1

            # now, for each x, we increment the corresponding entry
            def bin_to_base10(bits):
                return sum([2**i for i, b in enumerate(bits) if b == 1])

            for i in range(output.size(0)):
                current_x = (
                    current_xs[i].cpu().numpy().tolist()
                )  # binary repr of x (between 0 and 1024)
                current_x_base10 = bin_to_base10(current_x)
                hidden_layer_stats[layer]["hist_tandx"][
                    tuple(li[i]) + (current_x_base10,)
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

    # now we can compute the mutual information
    # I(T; Y) = sum_t sum_y p(t, y) log(p(t, y) / (p(t)p(y)))
    # p(y) = 0.5
    # I(T; X) = sum_t sum_x p(t, x) log(p(t, x) / (p(t)p(x)))
    # p(x) = 1 / 1024
    mis_y = dict()
    mis_x = dict()
    for layer, info in hidden_layer_stats.items():
        mi = 0
        for k, v in info["p_tandy"].items():
            t = k[:-1]
            mi += v * math.log(v / (info["p_t"][t] * 0.5))
        mis_y[layer] = mi

        mi = 0
        for k, v in info["p_tandx"].items():
            t = k[:-1]
            mi += v * math.log(v / (info["p_t"][t] * (1 / 1024)))
        mis_x[layer] = mi

    for layer in model:
        layer._forward_hooks.clear()

    return mis_y, mis_x


def print_mi(model):
    mis_y, mis_x = get_mutual_info(model)
    # print layer name, mi(T; Y), mi(T; X)
    for mi_y, mi_x in zip(mis_y.items(), mis_x.items()):
        layer_key = list(mi_y[0].state_dict().keys())[0]
        print(f"{layer_key}: I(T; Y)={mi_y[1]}, I(T; X)={mi_x[1]}")


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.ToTensor()


def tensor_to_binary(t: torch.Tensor) -> torch.Tensor:
    return ((t.unsqueeze(-1) & (1 << torch.arange(9, -1, -1))) > 0).int()


tensor_to_binary_batched = torch.vmap(tensor_to_binary)


def generate_dataset(n):
    ngroups = 4
    group_labels = torch.zeros(ngroups)
    # half groups are 0, half are 1, randomly
    group_labels[torch.randint(0, ngroups, (ngroups // 2,))] = 1

    x = torch.randint(0, 1024, (n,))
    y = torch.zeros(n)

    # y is 1 if x is in a group with label 1
    for i in range(n):
        groupn = x[i] // (1024 // ngroups)
        y[i] = group_labels[groupn]

    # x as binary
    x = tensor_to_binary_batched(x)

    return x.float(), y.long()


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
        self.x, self.y = generate_dataset(n)
        self.x = self.x.float()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_dataset = BinaryDataset(50000)
test_dataset = BinaryDataset(10000)

train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearAndActivation(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


ident = lambda x: x
model = nn.Sequential(
    nn.Flatten(),
    LinearAndActivation(10, 8, nn.Tanh()),
    LinearAndActivation(8, 6, nn.Tanh()),
    LinearAndActivation(6, 2, ident),
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1000):
    if epoch % 10 == 0:
        print("Before epoch", epoch)
        print_mi(model)
        with torch.no_grad():
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Epoch {epoch}: Accuracy {correct / total}")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()


print_mi(model)
