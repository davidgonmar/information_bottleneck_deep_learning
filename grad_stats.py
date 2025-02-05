import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

plt.ion()


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


params = list(model.parameters())
num_layers = len([p for p in params if p.ndimension() > 1])

grad_means_layers = [[] for _ in range(num_layers)]
grad_stds_layers = [[] for _ in range(num_layers)]
colors = plt.cm.tab10(np.linspace(0, 1, num_layers))

plt.ion()

for epoch in range(4000):
    if epoch % 5 == 0:
        with torch.no_grad():
            total, correct = 0, 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"Epoch {epoch}: Accuracy {correct / total:.4f}")

    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=True
    )
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    output = model(images)
    loss = loss_fn(output, labels)
    loss.backward()

    for i, p in enumerate([p for p in params if p.ndimension() > 1]):
        layer_norm = p.norm()
        if layer_norm.item() == 0:
            layer_norm = torch.tensor(1.0, device=p.device)
        grad_mean = p.grad.abs().mean().item() / layer_norm.item()
        grad_std = p.grad.std().item() / layer_norm.item()
        grad_means_layers[i].append(grad_mean)
        grad_stds_layers[i].append(grad_std)

    optimizer.step()

    plt.clf()
    for i in range(num_layers):
        plt.plot(grad_means_layers[i], color=colors[i], label=f"Layer {i} Mean")
        plt.plot(
            grad_stds_layers[i], color=colors[i], linestyle="--", label=f"Layer {i} Std"
        )
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Gradient Value")
    plt.legend(loc="upper right", fontsize="x-small")
    plt.pause(0.01)

plt.ioff()
plt.show()
