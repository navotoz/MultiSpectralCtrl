# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path

import numpy as np

# %%
data_base_dir = Path(
    r"C:\Users\omriber\Documents\Thesis\MultiSpectralCtrl\download")
data_fname = "img1_good.npy"
data = np.load(Path(data_base_dir, data_fname))
data.shape

# %%
import matplotlib.pyplot as plt

filters = [None, 8, 9, 10, 11, 12]
fig, ax = plt.subplots(2, 3)
for ch, filter in enumerate(filters):
    ax[ch // 3, ch % 3].imshow(data[ch].mean(0), cmap="gray")
    ax[ch // 3, ch % 3].set_axis_off()
    if filter is not None:
        ax[ch // 3, ch % 3].set_title(f"{filter} " + "$\mu$m")
    else:
        ax[ch // 3, ch % 3].set_title("Pan-Chromatic")

plt.suptitle(f"nCaptures = {data.shape[1]}")
plt.show()
