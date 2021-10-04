from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def plot_gl_as_func_temp(meas, list_blackbody_temperatures, n_pixels_to_plot: int = 4):
    meas_ = meas.mean(1)

    pixels = list(product(range(meas.shape[-2]), range(meas.shape[-1])))
    np.random.shuffle(pixels)

    fig, axs = plt.subplots(n_pixels_to_plot // 2, 2, dpi=300,
                            tight_layout=True, sharex=True, sharey=True)
    for ax, (h_, w_) in zip(axs.ravel(), pixels):
        ax.scatter(list_blackbody_temperatures, meas_[:, h_, w_])
        ax.set_title(f'({h_},{w_})')
        ax.grid()
    plt.suptitle(f'GL as a function of BlackBody temperature')
    fig.supxlabel('BlackBody Temperature [C]')
    fig.supylabel('Gray Levels [14Bit]')
    plt.locator_params(axis="x", nbins=len(list_blackbody_temperatures))
    plt.locator_params(axis="y", nbins=8)
    plt.show()
    plt.close()


def plot_regression_p_vs_p(list_power_panchormatic, est_power_panchromatic, n_pixels_to_plot: int = 4):
    pixels = list(product(range(
        est_power_panchromatic.shape[-2]), range(est_power_panchromatic.shape[-1])))
    np.random.shuffle(pixels)
    if len(est_power_panchromatic.shape) == 4:
        est = est_power_panchromatic.mean(1)
    else:
        est = est_power_panchromatic

    fig, axs = plt.subplots(n_pixels_to_plot // 2, 2, dpi=300,
                            tight_layout=True, sharex=True, sharey=True)
    for ax, (h_, w_) in zip(axs.ravel(), pixels):
        ax.plot(list_power_panchormatic,
                est[:, h_, w_], c='r', label='Estimation', linewidth=1)
        ax.scatter(list_power_panchormatic,
                   est[:, h_, w_], c='r', marker='X', s=5)
        ax.plot(list_power_panchormatic, list_power_panchormatic,
                c='b', label='Model', linewidth=0.7)
        ax.set_title(f'Pixel ({h_},{w_})')
        ax.legend(prop={'size': 5})
        ax.grid()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)

    plt.suptitle(f'Estimated Power as a function of the model Power')
    fig.supxlabel('Power [W??]')
    fig.supylabel('Power [W??]')
    plt.locator_params(axis="x", nbins=len(list_power_panchormatic) + 1)
    plt.locator_params(axis="y", nbins=8)
    plt.show()
    plt.close()


def plot_regression_diff(list_power_panchormatic, est_power_panchromatic, n_pixels_to_plot: int = 4):
    pixels = list(product(range(
        est_power_panchromatic.shape[-2]), range(est_power_panchromatic.shape[-1])))
    np.random.shuffle(pixels)
    if len(est_power_panchromatic.shape) == 4:
        est = est_power_panchromatic.mean(1)
    else:
        est = est_power_panchromatic

    diff = est.copy()
    for idx, t in enumerate(list_power_panchormatic):
        diff[idx] -= t

    fig, axs = plt.subplots(n_pixels_to_plot // 2, 2, dpi=300,
                            tight_layout=True, sharex=True, sharey=True)
    for ax, (h_, w_) in zip(axs.ravel(), pixels):
        ax.scatter(list_power_panchormatic,
                   diff[:, h_, w_], c='r', marker='X', s=5)
        ax.set_title(f'Pixel ({h_},{w_})')
        ax.grid()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
    plt.suptitle(f'Difference between real and estiamted Power')
    fig.supxlabel('Power [W??]')
    fig.supylabel('Difference [W??]')
    plt.locator_params(axis="x", nbins=len(list_power_panchormatic) + 1)
    plt.locator_params(axis="y", nbins=8)
    plt.show()
    plt.close()


def showFacetImages(img_arr, label_class, labels, facet_col=0, facet_col_wrap=4, title=None):
    n_facets = img_arr.shape[facet_col]
    eff_col_wrap = facet_col_wrap if n_facets > facet_col_wrap else n_facets
    fig = px.imshow(img_arr, facet_col=facet_col,
                    facet_col_wrap=eff_col_wrap, color_continuous_scale='gray', title=title)
    # Set facet titles
    for i in range(len(labels)):
        fig.layout.annotations[i]['text'] = f'{label_class} = {labels[i]}'
    fig.show()


def plotGlAcrossFrames(meas: np.ndarray, pix_idx: np.ndarray = None, wavelength: int = 0, title: str = '',
                       save_path: str = None, to_average: bool = False):
    """Plot the grey-level of a pixel across all frames, assuming the first
    dimension of the measurement is the number of frames, and the rest are the
    spatial dimensions

        Parameters:
            meas: the array containing the raw data of the measurements
            pix_idx: an nx2 array, where each row stands for a single pixel indices to be plotted.
                    If None - random choice.
            wavelength: The central wavelength of the BPF. If no filter - 0.
            title: if given, the title of the plot. Else, the default title is set.
            save_path: if given, saves the figure before displaying.
            to_average: if true, averages the value of the pixels.
    """
    if pix_idx is None:  # choose 4 random pixels at random
        pix_idx = np.random.randint(
            low=[0, 0], high=meas.shape[1:], size=(4, 2))
    grey_levels = meas[:, pix_idx[:, 0], pix_idx[:, 1]]
    if to_average:
        grey_levels = grey_levels.mean(-1)
    x = np.arange(len(grey_levels)) / 3600
    plt.figure(figsize=(16, 9))
    plt.plot(x, grey_levels, label=pix_idx if not to_average else None, linewidth=1)
    if not title:
        stmt = "Random Pixel Grey-Levels During Continuous Acquisition\n"
        if wavelength != 0:
            stmt += f'Filter {wavelength}nm'
        else:
            stmt += f'Pan-Chromatic'
    plt.title(title)
    plt.xticks(np.linspace(0, x[-1], len(set(x.astype('int'))), dtype=int))
    plt.xlabel("Time [Minutes]")
    plt.ylabel("Grey-Level")
    plt.grid()
    plt.legend() if not to_average else None
    if save_path:
        plt.savefig(save_path, transparent=False)
    return grey_levels
