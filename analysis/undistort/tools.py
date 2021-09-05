import os
import warnings
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from tqdm import tqdm


def calcRxPower(temperature: float, central_wl: int, bw: int = 1000, *, is_ideal_filt: bool = False, debug=False):
    """Calculates the power emmited by the the black body, according to Plank's law of black-body radiation, 
    after being filtered by the applied narrow-banded spectral filter.
    
        Parameters:
            temperature - the temperatureerature of the black body [C]
            central_wl - the central wavelength of the filter [nm]
            bw - the bandwidth of the filter [nm]
            is_ideal_filt - set to True to use an optimal rectangular filter centered about the central wavelength. 
            Otherwise, use the practical filter according to Thorlab's characterization.
    """
    # todo: use TAU's spec to asses the natural band-width of the camera for improving pan-chromatic and filtered calculations

    # # constants:
    KB = 1.380649e-23  # [Joule/Kelvin]
    H = 6.62607004e-34  # [m^2*kg/sec]
    C = 2.99792458e8  # [m/sec]

    # spectral radiance (Plank's function)
    bb = lambda lamda, T: 2 * H * C ** 2 / np.power(lamda, 5) * 1 / (np.exp(H * C / (KB * T * lamda)) - 1)

    # Celsuis to Kelvin:
    c2k = lambda T: T + 273.15
    k2c = lambda T: T - 273.15

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
        print(f"Estimated temperature (by integrating over plank's function): {T_hat:.4f}")

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

        if bw == np.inf:
            fig.suptitle("Pan-Chromatic")
        else:
            fig.suptitle(f"{central_wl} nm")
        plt.show()

    return (plank_filt * d_lambda).sum()


def load_npy_into_dict(path_to_files: Path):
    dict_measurements = {}

    paths = list(path_to_files.glob('*.npy'))
    for path in tqdm(paths, desc="Load measurements"):
        temperature_blackbody = int(path.stem.split('_')[-1])
        try:
            meas = np.load(str(path))
        except ValueError:
            print(f'Cannot load file {str(path)}')
            continue
        list_filters = sorted(pd.read_csv(path.with_suffix('.csv')).to_numpy()[:, 1])
        for idx, filter_name in enumerate(list_filters):
            dict_measurements.setdefault(temperature_blackbody, {}).setdefault(filter_name, meas[idx])
    return {k: dict_measurements[k] for k in sorted(dict_measurements.keys())}


def get_panchromatic_meas(path_to_files: Path):
    dict_measurements = load_npy_into_dict(path_to_files)
    list_power_panchormatic = [calcRxPower(temperature=t_bb, central_wl=0, bw=np.inf, is_ideal_filt=True, debug=False)
                               for t_bb in dict_measurements.keys()]
    return np.stack([dict_measurements[t_bb][0] for t_bb in dict_measurements.keys()]), \
           list_power_panchormatic, list(dict_measurements.keys())


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
    pixels = list(product(range(est_power_panchromatic.shape[-2]), range(est_power_panchromatic.shape[-1])))
    np.random.shuffle(pixels)
    if len(est_power_panchromatic.shape) == 4:
        est = est_power_panchromatic.mean(1)
    else:
        est = est_power_panchromatic

    fig, axs = plt.subplots(n_pixels_to_plot // 2, 2, dpi=300,
                            tight_layout=True, sharex=True, sharey=True)
    for ax, (h_, w_) in zip(axs.ravel(), pixels):
        ax.plot(list_power_panchormatic, est[:, h_, w_], c='r', label='Estimation', linewidth=1)
        ax.scatter(list_power_panchormatic, est[:, h_, w_], c='r', marker='X', s=5)
        ax.plot(list_power_panchormatic, list_power_panchormatic, c='b', label='Model', linewidth=0.7)
        ax.set_title(f'Pixel ({h_},{w_})')
        ax.legend(prop={'size': 5})
        ax.grid()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)

    plt.suptitle(f'Estimated Power as a function of the model Power')
    fig.supxlabel('Power [W??]')
    fig.supylabel('Power [W??]')
    plt.locator_params(axis="x", nbins=len(list_power_panchormatic)+1)
    plt.locator_params(axis="y", nbins=8)
    plt.show()
    plt.close()


def plot_regression_diff(list_power_panchormatic, est_power_panchromatic, n_pixels_to_plot: int = 4):
    pixels = list(product(range(est_power_panchromatic.shape[-2]), range(est_power_panchromatic.shape[-1])))
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
        ax.scatter(list_power_panchormatic, diff[:, h_, w_], c='r', marker='X', s=5)
        ax.set_title(f'Pixel ({h_},{w_})')
        ax.grid()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
    plt.suptitle(f'Difference between real and estiamted Power')
    fig.supxlabel('Power [W??]')
    fig.supylabel('Difference [W??]')
    plt.locator_params(axis="x", nbins=len(list_power_panchormatic)+1)
    plt.locator_params(axis="y", nbins=8)
    plt.show()
    plt.close()

