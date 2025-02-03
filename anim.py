import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

npmat = np.loadtxt("mis.csv", delimiter=",")

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

cmap = plt.colormaps["coolwarm"]
norm = plt.Normalize(vmin=0, vmax=npmat.shape[0] - 1)

scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig.colorbar(sm, ax=ax, label="Epoch (i)")

ax.set_xlabel("I(T; X)")
ax.set_ylabel("I(T; Y)")
ax.set_title("Mutual Information Matrix (Slide)")

ax.set_xlim(np.min(npmat[:, 1::2]), np.max(npmat[:, 1::2]))
ax.set_ylim(np.min(npmat[:, 0::2]), np.max(npmat[:, 0::2]))

ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
epoch_slider = Slider(ax_slider, "Epoch", 0, npmat.shape[0] - 1, valinit=0, valstep=1)


def update(epoch):
    epoch = int(epoch)
    start_epoch = max(0, epoch - 5)
    end_epoch = min(npmat.shape[0], epoch + 5)

    x_data = np.concatenate([npmat[i, 1::2] for i in range(start_epoch, end_epoch)])
    y_data = np.concatenate([npmat[i, 0::2] for i in range(start_epoch, end_epoch)])
    colors = np.concatenate(
        [np.full_like(npmat[i, 1::2], i) for i in range(start_epoch, end_epoch)]
    )

    scat.set_offsets(np.c_[x_data, y_data])
    scat.set_array(colors)
    ax.set_title(f"Mutual Information Matrix - Epochs {start_epoch} to {end_epoch - 1}")
    fig.canvas.draw_idle()


epoch_slider.on_changed(update)

update(0)

plt.show()
