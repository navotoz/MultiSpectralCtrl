import os
import warnings
from enum import Enum
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


class FilterWavelength(Enum):
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
                                       "FiltersResponse.xlsx"), sheet_name=str(central_wl),
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
        path, temperature_blackbody, filter_name = get_temperature_and_wavelength(path)
        meas = np.load(str(path))
    except ValueError:
        print(f'Cannot load file {str(path)}')
        return None
    return meas, temperature_blackbody, filter_name


def get_temperature_and_wavelength(path: (Path, str)):
    data = Path(path).stem.split('_')
    temperature_blackbody = int(data[-3])
    filter_name = int(data[-1])
    return path, temperature_blackbody, filter_name


def get_meas(path_to_files: (Path, str), filter_wavelength: FilterWavelength, *, omit_ops: Iterable = None):
    """Get the measurements acquired by the FLIR LWIR camera using a specific
        filter 

        Parameters: path_to_files: the path to the source files containing the
            data. ommit_op: a list of the operating-points to ommit as part of
            the prefiltering. e.g: [20, 40, 50] 
    """

    paths = [get_temperature_and_wavelength(p) for p in Path(path_to_files).glob('*.npy')]
    paths = filter(lambda x: x[-1] == filter_wavelength.value, paths)  # only load meas with the given filter
    paths = [p[0] for p in paths]

    # multithreading loading
    with Pool(8) as pool:
        list_meas = list(tqdm(pool.imap(func=load_npy, iterable=paths), total=len(paths), desc="Load measurements"))
    list_meas = list(filter(lambda x: x is not None, list_meas))

    # list into dict
    dict_measurements = {}
    while list_meas:
        meas, t_bb, filter_name = list_meas.pop()
        dict_measurements.setdefault(t_bb, meas)

    # sort nested dict keys
    dict_measurements = {k: dict_measurements[k] for k in sorted(dict_measurements.keys())}

    # remove invalid operating-points:
    if omit_ops is None:
        omit_ops = []
    for op in omit_ops:
        dict_measurements.pop(op)

    list_power_panchromatic = [calc_rx_power(temperature=t_bb) for t_bb in dict_measurements.keys()]
    list_blackbody_temperatures = list(dict_measurements.keys())
    return np.stack(list(dict_measurements.values())), list_power_panchromatic, list_blackbody_temperatures


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