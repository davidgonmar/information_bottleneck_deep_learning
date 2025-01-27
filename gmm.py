import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import os
from scipy.stats import multivariate_normal


class GMMDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples, n_features=2, n_classes=3, random_seed=42):
        np.random.seed(random_seed)
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

        self.means = np.random.uniform(-5, 5, size=(n_classes, n_features))
        self.covariances = np.array([np.eye(n_features) for _ in range(n_classes)])
        self.weights = np.random.dirichlet([1] * n_classes)

        self.x = []
        self.y = []
        for _ in range(n_samples):
            component = np.random.choice(n_classes, p=self.weights)
            sample = np.random.multivariate_normal(
                self.means[component], self.covariances[component]
            )
            self.x.append(sample)
            self.y.append(component)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_dataset = GMMDataset(n_samples=50000, n_features=10, n_classes=6)
test_dataset = GMMDataset(n_samples=2000, n_features=10, n_classes=6)


test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


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

    gmm_weights = train_dataset.weights
    gmm_means = train_dataset.means
    gmm_covs = train_dataset.covariances

    def compute_px(x):
        """Compute p(x) for a given x using the GMM parameters."""
        px = 0
        for weight, mean, cov in zip(gmm_weights, gmm_means, gmm_covs):
            px += weight * multivariate_normal.pdf(x, mean, cov)
        return px

    mis_y = dict()
    mis_x = dict()
    for layer, info in hidden_layer_stats.items():
        mi = 0
        for k, v in info["p_tandy"].items():
            t = k[:-1]
            y = k[-1]
            py = gmm_weights[y]  # p(y) from GMM
            mi += v * math.log(v / (info["p_t"][t] * py))
        mis_y[layer] = mi

        mi = 0
        for k, v in info["p_tandx"].items():
            t = k[:-1]
            x = np.array(k[-1])
            px = compute_px(x)  # p(x) from GMM
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
    LinearAndActivation(train_dataset.n_features, 5, nn.Tanh()),
    LinearAndActivation(5, 3, nn.Tanh()),
    LinearAndActivation(3, train_dataset.n_classes, ident),
).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Training loop with mutual information computation
mis = []

for epoch in range(500):
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
