import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os


def tensor_to_binary(t: torch.Tensor) -> torch.Tensor:
    return (
        (t.unsqueeze(-1) & (1 << torch.arange(11, -1, -1, device=t.device))) > 0
    ).int()


tensor_to_binary_batched = torch.vmap(tensor_to_binary)


def generate_dataset(n, group_labels, max_int=4096):
    """
    n: number of samples
    group_labels: a tensor of shape (ngroups,) that gives the label (0 or 1) for each group
    max_int: the maximum integer (exclusive) from which to sample.
    """
    ngroups = group_labels.numel()
    x = torch.randint(0, max_int, (n,))
    y = torch.zeros(n, dtype=torch.long)

    group_size = max_int // ngroups
    for i in range(n):
        groupn = x[i] // group_size
        y[i] = group_labels[groupn]

    x_binary = tensor_to_binary_batched(x)

    return x_binary.float(), y


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, n, group_labels, max_int=4096):
        self.n = n
        self.x, self.y = generate_dataset(n, group_labels, max_int)
        self.n_features = self.x.size(1)
        self.n_classes = 2

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_datasets(n_train=50000, n_test=10000, ngroups=16, max_int=4096):
    """
    Returns the training and test datasets.
    A single random mapping (group_labels) is created and used for both.
    """

    group_labels = torch.zeros(ngroups, dtype=torch.long)
    indices = torch.randperm(ngroups)[: (ngroups // 2)]
    group_labels[indices] = 1

    train_dataset = BinaryDataset(n_train, group_labels, max_int)
    test_dataset = BinaryDataset(n_test, group_labels, max_int)
    return train_dataset, test_dataset


train_dataset, test_dataset = get_datasets()

train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)


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
            py = 1 / train_dataset.n_classes
            mi += v * math.log(v / (info["p_t"][t] * py))
        mis_y[layer] = mi

        mi = 0
        for k, v in info["p_tandx"].items():
            t = k[:-1]
            px = 1 / (2**train_dataset.n_features)
            mi += v * math.log(v / (info["p_t"][t] * px))
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

    return mis_y, mis_x


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
    LinearAndActivation(train_dataset.n_features, 6, nn.Tanh()),
    LinearAndActivation(6, 4, nn.Tanh()),
    LinearAndActivation(4, train_dataset.n_classes, ident),
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()


mis = []

for epoch in range(800):
    if epoch % 5 == 0:
        print("Before epoch", epoch)
        mis.append(print_mi(model))
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

    images, labels = next(
        iter(DataLoader(train_dataset, batch_size=100, shuffle=True))
    )  # random batch
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(images)
    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()

print("Saving mutual informations along training")
mis2 = []


def flatten(lis):
    res = []
    for l in lis:
        if isinstance(l, (list, tuple)):
            res.extend(flatten(l))
        else:
            res.append(l)
    return res


for mis_y, mis_x in mis:
    mis2.append(flatten([*zip(list(mis_y.values()), list(mis_x.values()))]))

npmat = np.array(mis2)

if os.path.exists("mis.csv"):
    os.remove("mis.csv")
np.savetxt("mis.csv", npmat, delimiter=",")
print("Done")
