from itertools import product
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from IPython.display import Latex, display
from sklearn.linear_model import LinearRegression

from tools import (FilterWavelength, choose_random_pixels,
                   get_blackbody_temperature_from_path, get_measurements)


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


def plot_gl(*, grey_levels: np.ndarray, save_path: str = None, to_average: bool = False):
    """Plot the grey-level of a pixel across all frames, assuming the first
    dimension of the measurement is the number of frames, and the rest are the
    spatial dimensions

        Parameters:
            grey_levels:
            save_path: if given, saves the figure before displaying.
            to_average: if true, averages the value of the pixels.
    """

    if to_average:
        grey_levels = grey_levels.mean(-1)
    x = np.arange(len(grey_levels)) / 3600
    plt.figure(figsize=(16, 9))
    plt.plot(x, grey_levels, linewidth=1)
    plt.title("Random Pixel Grey-Levels During Continuous Acquisition")
    plt.xticks(np.linspace(0, x[-1], len(set(x.astype('int'))), dtype=int))
    plt.xlabel("Time [Minutes]")
    plt.ylabel("Grey-Level")
    plt.grid()
    if save_path:
        plt.savefig(save_path, transparent=False)
    return grey_levels


def plot_gl_and_camera_temperature(*, grey_levels: np.ndarray, housing: np.ndarray, fpa: np.ndarray):
    x_time = np.array(range(grey_levels.shape[0])) / 3600
    gl = grey_levels[:, 1]

    fig, ax = plt.subplots(figsize=(16, 9))

    twin_housing = ax.twinx()
    twin_fpa = ax.twinx()

    # Offset the right spine of twin_housing.  The ticks and label have already been
    # placed on the right by twinx above.
    twin_housing.spines.right.set_position(("axes", 1.08))

    p1, = ax.plot(x_time, gl, "b-", label="Grey Levels")
    p2, = twin_housing.plot(x_time, housing, "r-", label="Housing [C]")
    p3, = twin_fpa.plot(x_time, fpa, "g-", label="FPA [C]")

    ax.set_ylabel("Grey Levels")
    ax.set_xlabel("Time [Minutes]")
    twin_housing.set_ylabel('Housing [C]')
    twin_fpa.set_ylabel('FPA [C]')

    ax.yaxis.label.set_color(p1.get_color())
    twin_housing.yaxis.label.set_color(p2.get_color())
    twin_fpa.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin_housing.tick_params(axis='y', colors=p2.get_color(), **tkw)
    twin_fpa.tick_params(axis='y', colors=p3.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw)

    ax.legend(handles=[p1, p2, p3])

    ax.set_title("Grey Level and Camera's Temperature along different frames")
    plt.grid()
    plt.show()
    plt.close()


def plot_regression_gl_to_t(*, grey_levels: np.ndarray, fpa: np.ndarray,):
    x, y = fpa[:, None], grey_levels[:, 0][:, None]
    lr = LinearRegression().fit(x, y)
    plt.figure(figsize=(16, 9))
    res = sns.regplot(x=x, y=y, fit_reg=True)
    plt.title(f"Regression model for grey-level Vs temperature")
    plt.xlabel("Temperature[C]")
    plt.ylabel("Grey-Level")
    plt.grid()
    plt.show()
    plt.close()
    sleep(0.5)

    display(Latex(
        fr"Regression result: $GL = {lr.coef_.squeeze():.3f} \times T +{lr.intercept_.squeeze():.3f}$"))
    display(Latex(fr"$R^2 = {lr.score(x, y):.3f}$"))
    print("Correlation Coefficients:")
    print(np.corrcoef(x.squeeze(), y.squeeze()))
    print()


def plot_tlinear_effect(grey_levels: np.ndarray, temperature_blackbody: int):
    x_time = np.array(range(grey_levels.shape[0])) / 3600
    t = 0.04 * grey_levels - 273.15

    plt.figure(figsize=(16, 9))
    plt.plot(x_time, t, label='Camera estimation [C]')
    plt.plot(x_time, np.ones_like(x_time) * temperature_blackbody, '--b', linewidth=2,
             label='BlackBody temperature [C]')
    plt.legend()
    plt.title(f'Camera temperature estimation')
    plt.xlabel('Time [Minutes]')
    plt.ylabel('Temperature [C]')
    plt.yticks(np.linspace(t.min(), t.max(), 15))
    plt.xticks(np.arange(x_time.min(), x_time.max(), 2))
    plt.grid()
    plt.show()

    plt.figure(figsize=(16, 9))
    plt.plot(x_time, t - temperature_blackbody)
    plt.plot(x_time, np.zeros_like(x_time), '--b', linewidth=2)
    plt.title(f'Camera temperature estimation Error\nBlackBody temperature {temperature_blackbody}C')
    plt.xlabel('Time [Minutes]')
    plt.ylabel('Temperature difference [C]')
    plt.yticks(np.linspace((t - temperature_blackbody).min(), (t - temperature_blackbody).max(), 15))
    plt.xticks(np.arange(x_time.min(), x_time.max(), 2))
    plt.grid()
    plt.show()


def plot_regression(x, y, deg=1, xlabel=None, ylabel=None, title=None):
    coeffs = np.polyfit(x, y, deg)
    plt.figure(figsize=(16, 9))
    sns.regplot(x=x, y=y, order=deg)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()

    n_coeffs = deg+1
    regression_parts = [rf"{coeffs[k]:.3f} \times T^{deg-k} +" for k in range(n_coeffs) if k < deg]
    regression_parts.append(f"{coeffs[-1]:.3f}")
    regression_formula = f"$GL = {' '.join(regression_parts)}$".replace("T^1", "T")
    # TODO: finalize regression formula in display
    display(Latex(
        fr"Regression result: {regression_formula}"))
    return coeffs

def plot_rand_pix_regress(x_vals, meas, deg=1, xlabel=None):
    i, j = choose_random_pixels(1, meas.shape[2:])
    x, y = np.array(x_vals).repeat(meas.shape[1]), meas[..., i, j].flatten()
    res = plot_regression(x, y, deg, xlabel, ylabel="Grey-Level", title=f"Regression model for random pixel {(i[0], j[0])}")
    return res



def load_and_plot(path: (str, Path)):
    path = Path(path)
    path, temperature_blackbody = get_blackbody_temperature_from_path(path)
    grey_levels, fpa, housing, _, _, camera_parameters = get_measurements(path_to_files=path,
                                                                          filter_wavelength=FilterWavelength.PAN)
    pix_idx = choose_random_pixels(n_pixels=4, img_dims=grey_levels.shape[-2:])
    grey_levels = grey_levels[:, pix_idx[:, 0], pix_idx[:, 1]]
    camera_parameters = camera_parameters[0]
    sleep(0.5)

    print(f"\nFFC {camera_parameters['ffc_period'] != 0}\t TLinear {camera_parameters['tlinear'] == 1}", flush=True)
    plot_gl(grey_levels=grey_levels)
    plot_gl_and_camera_temperature(grey_levels=grey_levels, housing=housing, fpa=fpa)
    plot_regression_gl_to_t(grey_levels=grey_levels, fpa=fpa)
    if camera_parameters['tlinear']:
        plot_tlinear_effect(grey_levels, temperature_blackbody)
    print('#' * 80 + '\n' + '#' * 80 + '\n', flush=True)
