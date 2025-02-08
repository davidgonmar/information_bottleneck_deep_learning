import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os
from ib import get_mutual_info
import argparse


def tensor_to_binary(t: torch.Tensor) -> torch.Tensor:
    return (
        (t.unsqueeze(-1) & (1 << torch.arange(11, -1, -1, device=t.device))) > 0
    ).int()


tensor_to_binary_batched = torch.vmap(tensor_to_binary)

parser = argparse.ArgumentParser()
parser.add_argument("--activation", type=str, default="tanh")
args = parser.parse_args()

torch.manual_seed(42)
ACTIVATION_TO_BIN_CONFIG = {
    "tanh": (-1, 1, 30),
    "relu": (0, 1, 30),
    "sigmoid": (0, 1, 30),
    "identity": (-1, 1, 30),
    "softmax": (0, 1, 30),
}


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
        self.n_features: int = self.x.size(1)
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


p_y = lambda y: 1 / 2
p_x = lambda x: 1 / (2**12)


def get_bin_config(layer):
    assert isinstance(layer, LinearAndActivation)
    if isinstance(layer.activation, nn.Tanh):
        return ACTIVATION_TO_BIN_CONFIG["tanh"]
    elif isinstance(layer.activation, nn.ReLU):
        return ACTIVATION_TO_BIN_CONFIG["relu"]
    elif isinstance(layer.activation, nn.Sigmoid):
        return ACTIVATION_TO_BIN_CONFIG["sigmoid"]
    elif isinstance(layer.activation, nn.Identity):
        return ACTIVATION_TO_BIN_CONFIG["identity"]
    elif isinstance(layer.activation, nn.Softmax):
        return ACTIVATION_TO_BIN_CONFIG["softmax"]
    elif isinstance(layer.activation, nn.LogSoftmax):
        return ACTIVATION_TO_BIN_CONFIG["softmax"]

    raise ValueError("Activation not supported: ", layer.activation)


def print_mi(model):
    mis_y, mis_x = get_mutual_info(
        model, test_loader, [LinearAndActivation], p_x, p_y, get_bin_config
    )
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


act = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "identity": nn.Identity(),
}[args.activation]

model = nn.Sequential(
    nn.Flatten(),
    LinearAndActivation(train_dataset.n_features, 8, act),
    LinearAndActivation(8, 6, act),
    LinearAndActivation(6, 4, act),
    LinearAndActivation(4, train_dataset.n_classes, nn.Softmax(dim=1)),
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.NLLLoss()


mis = []

for epoch in range(2000):
    if epoch % 10 == 0:
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

    idxs = torch.randint(0, len(train_dataset), (1000,))
    images, labels = train_dataset[idxs]
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    output = torch.log(model(images))  # log softmax
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
