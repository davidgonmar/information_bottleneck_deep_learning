import numpy as np
import matplotlib.pyplot as plt

npmat = np.loadtxt("mis.csv", delimiter=",")

cmap = plt.colormaps["coolwarm"]


norm = plt.Normalize(vmin=0, vmax=npmat.shape[0] - 1)

fig, ax = plt.subplots()


for i in range(npmat.shape[0]):
    epoch_coords = []

    for j in range(npmat.shape[1] // 2):
        x = npmat[i, j * 2 + 1]
        y = npmat[i, j * 2]

        ax.plot(x, y, "o", color=cmap(norm(i)), markersize=3)
        epoch_coords.append((x, y))

    epoch_coords = np.array(epoch_coords)
    ax.plot(
        epoch_coords[:, 0],
        epoch_coords[:, 1],
        color="gray",
        linestyle="-",
        linewidth=0.2,
        alpha=0.5,
    )

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label="Epoch (i)")

ax.set_xlabel("I(T; X)")
ax.set_ylabel("I(T; Y)")
ax.set_title("Mutual Information Matrix")

plt.show()
