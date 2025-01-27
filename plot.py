import numpy as np
import matplotlib.pyplot as plt

npmat = np.loadtxt("mis.csv", delimiter=",")

cmap = plt.colormaps["coolwarm"]
norm = plt.Normalize(vmin=0, vmax=npmat.shape[0] - 1)

fig, ax = plt.subplots()

for i in range(npmat.shape[0]):
    for j in range(npmat.shape[1] // 2):
        ax.plot(npmat[i, j * 2 + 1], npmat[i, j * 2], "o", color=cmap(norm(i)))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label="Epoch (i)")

ax.set_xlabel("I(T; X)")
ax.set_ylabel("I(T; Y)")
ax.set_title("Mutual Information Matrix")

plt.show()
