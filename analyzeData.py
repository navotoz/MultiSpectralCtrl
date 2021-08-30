# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from pathlib import Path

# %%
data_base_dir= r"C:\Users\omriber\Documents\Thesis\MultiSpectralCtrl\Downloads"
data_fname = "20210823_h19m51s58.npy"
data = np.load(Path(data_base_dir, data_fname), allow_pickle=True)
data.shape


# %%
import matplotlib.pyplot as plt

filters = [None, 8, 9, 10, 11, 12]
fig, ax = plt.subplots(2, 3)
for ch, filter in enumerate(filters):
    ax[ch//3, ch%3].imshow(data[ch].mean(0), cmap="gray")
    ax[ch//3, ch%3].set_axis_off()
    if filter is not None:
        ax[ch//3, ch%3].set_title(f"{filter} " + "$\mu$m")
    else:
        ax[ch//3, ch%3].set_title("Pan-Chromatic")

plt.suptitle(f"nCaptures = {data.shape[1]}")
plt.show()

