from multiprocessing.dummy import Pool

import plotly.express as px
import os
import warnings
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class SpectralFilter(Enum):
    """ An enumeration for Flir's LWIR filters, where the keys and values are
    the cenrtal frequencies of the filters in nano-meter."""
    PAN = 0
    nm8000 = 8000
    nm9000 = 9000
    nm10000 = 10000
    nm11000 = 11000
    nm12000 = 12000


def calc_rx_power(temperature: float, central_wl: int = 10500, bw: int = 3000, *, is_ideal_filt: bool = True,
                  debug=False):
    """Calculates the power emmited by the the black body, according to Plank's
    law of black-body radiation, after being filtered by the applied
    narrow-banded spectral filter. Default parameters fit the pan-chromatic
    (non-filtered) case.

        Parameters:
            temperature - the temperatureerature of the black body [C]
            central_wl - the central wavelength of the filter [nm]
            bw - the bandwidth of the filter [nm]
            is_ideal_filt - set to True to use an optimal rectangular filter centered about the central wavelength. 
            Otherwise, use the practical filter according to Thorlab's characterization.
    """
    # todo: use TAU's spec to asses the natural band-width of the camera for
    # improving pan-chromatic and filtered calculations

    # # constants:
    KB = 1.380649e-23  # [Joule/Kelvin]
    H = 6.62607004e-34  # [m^2*kg/sec]
    C = 2.99792458e8  # [m/sec]

    # spectral radiance (Plank's function)
    def bb(lamda, T):
        return 2 * H * C ** 2 / np.power(lamda, 5) * \
               1 / (np.exp(H * C / (KB * T * lamda)) - 1)

    # Celsuis to Kelvin:
    def c2k(T):
        return T + 273.15

    def k2c(T):
        return T - 273.15

    if is_ideal_filt:  # assume ideal rectangular filter with 1000nm bandwidth and a constant amplification of 1
        d_lambda = 1e-9
        if bw == np.inf:  # temporarily used as an assumption for the pan-chromatic
            bp_filter = pd.Series(
                index=d_lambda * np.arange(2e4), data=np.ones(int(2e4)))
        else:
            bp_filter = pd.Series(index=d_lambda * np.arange(
                central_wl - bw // 2, central_wl + bw // 2), data=np.ones(bw))

    else:
        # load filter from xlsx:
        bp_filter = pd.read_excel(Path(os.path.dirname(__file__),
                                       "../FiltersResponse.xlsx"), sheet_name=str(central_wl),
                                  engine='openpyxl', index_col=0, usecols=[0, 1]).dropna().squeeze()

        # remove entries Bw far away from the central frequency:
        below_bw = bp_filter.index.values < (central_wl - bw) * 1e-3
        above_bw = bp_filter.index.values > (central_wl + bw) * 1e-3
        inside_bw = np.bitwise_not(np.bitwise_or(below_bw, above_bw))
        bp_filter = bp_filter[inside_bw]

        # convert wavelength to meters and get delta-lambda vector:
        bp_filter.index *= 1e-6
        d_lambda = np.diff(bp_filter.index.values)
        d_lambda = np.append(d_lambda, d_lambda[-1])

        # convert Transmission from percent to normalized values:
        bp_filter /= 100

    lambda_grid = bp_filter.index.values  # [nm]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plank_unfilt = bb(lambda_grid, c2k(temperature))
    plank_filt = plank_unfilt * bp_filter

    if debug:
        import matplotlib.pyplot as plt
        base_grid = np.arange(2e4) * 1e-9  # [m]

        # validate integral over whole spectrum:
        valid_grid = np.arange(1e6) * 1e-9  # [m]
        sigma = 5.670373e-8  # [W/(m^2K^4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            L = np.nansum(bb(valid_grid, c2k(temperature)) * 1e-9)
        T_hat = k2c(np.power(L / sigma * np.pi, 1 / 4))
        print(f"Input temperature: {temperature:.4f}")
        print(
            f"Estimated temperature (by integrating over plank's function): {T_hat:.4f}")

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))

        # plot filter response:
        ax[0].plot(lambda_grid * 1e9, bp_filter * 100, label="filter Response")
        closest_idx = np.abs(bp_filter.index - central_wl * 1e-9).argmin()
        ax[0].vlines(central_wl, ymin=0, ymax=bp_filter.iloc[closest_idx] * 100, linestyles="dashed",
                     label="$\lambda_c$")
        ax[0].set_title("Band-Pass Filter Response")
        ax[0].set_ylabel("Transmitance [%]")
        ax[0].set_xlabel("$\lambda$[nm]")
        ax[0].grid()
        ax[0].legend()

        # plot filter vs whole spectrum:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ax[1].plot(base_grid * 1e9, bb(base_grid,
                                           c2k(temperature)), label="complete spectrum")

        ax[1].plot(lambda_grid * 1e9, plank_unfilt,
                   label="segment of interest")
        ax[1].plot(lambda_grid * 1e9, plank_filt, label="filtered spectrum")
        ax[1].set_ylabel("Spectral Radiance")
        ax[1].set_xlabel("$\lambda$[nm]")
        ax[1].grid()
        ax[1].legend()
        ax[1].set_title("The filtered segment of the spectral radiance")

        if central_wl == 10500 and bw == 3000:
            fig.suptitle(f"Pan-Chromatic at T={c2k(temperature)}[K]")
        else:
            fig.suptitle(f"{central_wl} nm Filter at T={c2k(temperature)}[K]")
        plt.show()

    return (plank_filt * d_lambda).sum()


def load_npy(path: Path):
    try:
        data = path.stem.split('_')
        temperature_blackbody = int(data[-3])
        filter_name = int(data[-1])
        meas = np.load(str(path))
    except ValueError:
        print(f'Cannot load file {str(path)}')
        return None
    return meas, filter_name, temperature_blackbody


def get_meas(path_to_files: Path, filter_obj: SpectralFilter = SpectralFilter.PAN, *, ommit_ops: Iterable = None):
    """Get the measurements acquired by the FLIR LWIR camera using a specific
        filter 

        Parameters: path_to_files: the path to the source files containing the
            data. ommit_op: a list of the operating-points to ommit as part of
            the prefiltering. e.g: [20, 40, 50] 
    """

    paths = list(path_to_files.glob('*.npy'))

    # multithreading loading
    with Pool(8) as pool:
        list_meas = list(tqdm(pool.imap(func=load_npy, iterable=paths), total=len(paths), desc="Load measurements"))
    list_meas = list(filter(lambda x: x is not None, list_meas))

    # list into dict
    dict_measurements = {}
    while list_meas:
        meas, filter_name, t_bb = list_meas.pop()
        dict_measurements.setdefault(t_bb, {}).setdefault(filter_name, meas)

    # sort nested dict keys
    for t_bb, v in dict_measurements.items():
        dict_measurements[t_bb] = {k: v[k] for k in sorted(v.keys())}
    dict_measurements = {k: dict_measurements[k] for k in sorted(dict_measurements.keys())}

    # remove invalid operating-points:
    if ommit_ops is None:
        ommit_ops = []
    for op in ommit_ops:
        dict_measurements.pop(op)

    list_power_panchromatic = [calc_rx_power(temperature=t_bb) for t_bb in dict_measurements.keys()]
    list_blackbody_temperatures = list(dict_measurements.keys())

    # leave only the required measurements, according to given filter
    for t_bb, v in dict_measurements.items():
        dict_measurements[t_bb] = v[filter_obj.value]
    return np.stack([dict_measurements[t_bb] for t_bb in sorted(dict_measurements.keys())]), \
           list_power_panchromatic, list_blackbody_temperatures


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


def prefilt_cam_meas(cam_meas: np.ndarray, *, first_valid_meas: int = 3, med_filt_sz: int = 2):
    """pre-filtering of raw measurements taken using Flir's TAU-2 LWIR camera.
        The filtering pipe is based on insights gained and explored in the
        'meas_inspection' notebook available under the same parent directory.

        Parameters: cam_meas: a 4D hypercube that contains the data collected by
        a set of measurements. first_valid_idx: the first valid measurement in
        all operating points. All measurements taken before that will be
        discarded. med_filt_sz: the size of the median filter used to clean dead
        pixels from the measurements.
    """
    from scipy.ndimage import median_filter
    cam_meas_valid = cam_meas[:, first_valid_meas:, ...]
    cam_meas_filt = median_filter(
        cam_meas_valid, size=(1, 1, med_filt_sz, med_filt_sz))
    return cam_meas_filt


def showFacetImages(img_arr, label_class, labels, facet_col=0, facet_col_wrap=4, title=None):
    n_facets = img_arr.shape[facet_col]
    eff_col_wrap = facet_col_wrap if n_facets > facet_col_wrap else n_facets
    fig = px.imshow(img_arr, facet_col=facet_col,
                    facet_col_wrap=eff_col_wrap, color_continuous_scale='gray', title=title)
    # Set facet titles
    for i in range(len(labels)):
        fig.layout.annotations[i]['text'] = f'{label_class} = {labels[i]}'
    fig.show()


def plotGlAcrossFrames(meas: np.ndarray, pix_idx: np.ndarray = None, wavelength: int = 0):
    """Plot the grey-level of a pixel across all frames, assuming the first
    dimension of the measurement is the number of frames, and the rest are the
    spatial dimensions
        
        Parameters:
            meas: the array containing the raw data of the measurements
            pix_idx: an nx2 array, where each row stands for a single pixel indices to be plotted.
                    If None - random choice.
            wavelength: The central wavelength of the BPF. If no filter - 0.
    
    """
    if pix_idx is None:  # choose 4 random pixels at random
        pix_idx = np.random.randint(
            low=[0, 0], high=meas.shape[1:], size=(4, 2))
    grey_levels = meas[:, pix_idx[:, 0], pix_idx[:, 1]]
    x = np.arange(len(grey_levels)) / 3600
    plt.figure(figsize=(16, 9))
    plt.plot(x, grey_levels, label=pix_idx, linewidth=1)
    stmt = "Random Pixel Grey-Levels During Continuous Acquisition\n"
    if wavelength != 0:
        stmt += f'Filter {wavelength}nm'
    else:
        stmt += f'Pan-Chromatic'
    plt.title(stmt)
    plt.xlabel("time[min]")
    plt.ylabel("Grey-Level")
    plt.grid()
    plt.legend()
    return grey_levels


def plotGlAcrossFramesPlotly(meas: np.ndarray, pix_idx: np.ndarray = None, wavelength: int = 0):
    """Plot the grey-level of a pixel across all frames, assuming the first
    dimension of the measurement is the number of frames, and the rest are the
    spatial dimensions

        Parameters:
            meas: the array containing the raw data of the measurements
            pix_idx: an nx2 array, where each row stands for a single pixel indices to be plotted.
                    If None - random choice.
            wavelength: The central wavelength of the BPF. If no filter - 0.

    """
    if pix_idx is None:  # choose 4 random pixels at random
        pix_idx = np.random.randint(low=[0, 0], high=meas.shape[1:], size=(4, 2))
    grey_levels = meas[:, pix_idx[:, 0], pix_idx[:, 1]]
    x = np.arange(len(grey_levels)) / 3600
    df = pd.DataFrame(columns=['time'], data=x.tolist())
    for idx in range(grey_levels.shape[-1]):
        df[str(pix_idx[idx])] = grey_levels[:, idx].tolist()
    stmt = "Random Pixel Grey-Levels During Continuous Acquisition\t\t"
    if wavelength != 0:
        stmt += f'Filter {wavelength}nm'
    else:
        stmt += f'Pan-Chromatic'
    fig = px.line(df, x='time', y=df.columns, title=stmt,
                  labels={"time": "Time [minutes]", "value": "Grey Levels", "variable": "Pixels [h,w]"}, )
    fig.show()
