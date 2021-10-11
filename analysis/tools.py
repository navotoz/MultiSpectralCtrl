import base64
import io
import os
import pickle
import warnings
from enum import Enum
from itertools import repeat
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
import scipy
from tqdm import tqdm
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class FilterWavelength(Enum):
    """ An enumeration for Flir's LWIR filters, where the keys and values are
    the cenrtal frequencies of the filters in nano-meter."""
    PAN = 0
    nm8000 = 8000
    nm9000 = 9000
    nm10000 = 10000
    nm11000 = 11000
    nm12000 = 12000


def calc_rx_power(temperature: float, filt: FilterWavelength = FilterWavelength.PAN, debug=False):
    """Calculates the power emmited by the the black body, according to Plank's
    law of black-body radiation, after being filtered by the applied
    narrow-banded spectral filter. Default parameters fit the pan-chromatic
    (non-filtered) case.

        Parameters:
            temperature: the temperatureerature of the black body [C]
            filter: the central wavelength of the filter [nm]
            bw: the bandwidth of the filter [nm]
            is_ideal_filt: set to True to use an optimal rectangular filter centered about the central wavelength. Otherwise, use the practical filter according to Thorlab's characterization.
    """
    # todo: use TAU's spec to asses the natural band-width of the camera for
    # improving pan-chromatic and filtered calculations

    # # constants:
    KB = 1.380649e-23  # [Joule/Kelvin]
    H = 6.62607004e-34  # [m^2*kg/sec]
    C = 2.99792458e8  # [m/sec]

    #
    def bb(lamda: float, T: float):
        """calculates spectral radiance (Plank's function)

            Parameters:
                lamda: the wavelength in meters
                T: the temperature in Kelvin
        """
        return 2 * H * C ** 2 / np.power(lamda, 5) * \
            1 / (np.exp(H * C / (KB * T * lamda)) - 1)

    # Celsuis to Kelvin:
    def c2k(T):
        return T + 273.15

    def k2c(T):
        return T - 273.15
    # %%
    # calculate lens response:

    lens_resp_raw = pd.read_excel(Path(os.path.dirname(__file__),
                                       "FiltersResponse.xlsx"), sheet_name="lens_680138-002",
                                  engine='openpyxl', index_col=0, usecols=[0, 1]).dropna().squeeze()
    # convert Transmission from percent to normalized values:
    lens_resp_raw /= 100

    # interpolate at regular grid with 1nm granularity:
    # TODO: complete lens model once lens responsivity is available from 7micro as well
    wl_band = ((lens_resp_raw.index[0] - 1) * 1000,
               lens_resp_raw.index[-1] * 1000)  # in [nm]
    # granularity in meters (will be used later on for the actual power calculation)
    d_lambda = 1e-9
    # grid with 1nm granularity
    wl_grid = np.arange(start=wl_band[0], stop=wl_band[-1], dtype=int)
    lens_resp = pd.Series(index=wl_grid, data=np.ones_like(wl_grid))

    # %%
    # calculate filter's response:

    if filt == FilterWavelength.PAN:
        # assume ideal rectangular filter with 3000nm bandwidth and a constant amplification of 1
        central_wl = (0.5 * (wl_band[0] + wl_band[1])).astype(int)
        filt_resp = pd.Series(index=wl_grid, data=np.ones_like(wl_grid))

    else:
        bw = 1000
        central_wl = filt.value
        # load filter from xlsx:
        filt_resp_raw = pd.read_excel(Path(os.path.dirname(__file__),
                                           "FiltersResponse.xlsx"), sheet_name=str(central_wl),
                                      engine='openpyxl', index_col=0, usecols=[0, 1]).dropna().squeeze()
        filt_resp_raw.index *= 1e3  # convert to [nm]

        # interpolate the values in the relevant grid:
        interp_model = interp1d(x=filt_resp_raw.index,
                                y=filt_resp_raw.values, kind="cubic")

        filt_resp = pd.Series(index=wl_grid, data=interp_model(wl_grid))

        # remove entries Bw far away from the central frequency:
        below_bw = filt_resp.index.values < (central_wl - bw)
        above_bw = filt_resp.index.values > (central_wl + bw)
        inside_bw = np.bitwise_not(np.bitwise_or(below_bw, above_bw))
        filt_resp = filt_resp[inside_bw]
        # convert Transmission from percent to normalized values:
        filt_resp /= 100

    # %%
    # calculate the equivalent system's transmittance to black-body radiation:

    plank = pd.Series(index=np.arange(1e6, dtype=int) ,dtype=float)  # [nm]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plank.loc[:] = bb(plank.index * d_lambda, c2k(temperature))

    trans_resp = lens_resp[filt_resp.index] * filt_resp
    plank_trans = plank.loc[filt_resp.index] * trans_resp

    # %%
    # Plot results:
    def rad_power(density, dl): return np.nansum(plank * d_lambda)
    if debug:
        import matplotlib.pyplot as plt

        # validate integral over whole spectrum:
        sigma = 5.670373e-8  # [W/(m^2K^4)]
        L = rad_power(plank, d_lambda)  # integrate over plank
        T_hat = k2c(np.power(L / sigma * np.pi, 1 / 4))
        print(f"Input temperature: {temperature:.4f}")
        print(
            f"Estimated temperature (by integrating over plank's function): {T_hat:.4f}")

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))

        # plot transmittance response:
        ax[0].plot(lens_resp * 100, label="Lense transmittance")
        if filt != FilterWavelength.PAN:
            ax[0].plot(filt_resp * 100,
                    label=f"{filt.value//1000} $\mu m$ filter transmittance")
        ax[0].plot(trans_resp * 100, label=f"combined transmittance")
        ax[0].set_title("Optical Transmittance")
        ax[0].set_ylabel("Transmittance [%]")
        ax[0].set_xlabel("$\lambda$[nm]")
        ax[0].grid()
        ax[0].legend()

        # plot raw spectrum and transmitted spectrum:
        ax[1].plot(plank, label="complete spectrum")
        ax[1].plot(wl_grid, plank.loc[wl_grid], label="TAU-2 bandwidth")
        ax[1].plot(plank_trans, label="transmitted spectrum")
        ax[1].set_ylabel("Spectral Radiance")
        ax[1].set_xlabel("$\lambda$[nm]")
        ax[1].grid()
        ax[1].legend()
        ax[1].set_title("The filtered segment of the spectral radiance")
        ax[1].set_xlim([0, 2e4])
        if filt == FilterWavelength.PAN:
            fig.suptitle(f"Pan-Chromatic at T={c2k(temperature)}[K]")
        else:
            fig.suptitle(f"{central_wl} nm Filter at T={c2k(temperature)}[K]")
        plt.show()

    return rad_power(plank_trans, d_lambda)


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
    cam_meas_valid = cam_meas[:, first_valid_meas:, ...]
    cam_meas_filt = median_filter(
        cam_meas_valid, size=(1, 1, med_filt_sz, med_filt_sz))
    return cam_meas_filt


def load_pickle(path: Path):
    try:
        path, temperature_blackbody = get_blackbody_temperature_from_path(path)
        with open(str(path), 'rb') as fp:
            meas = pickle.load(fp)
    except ValueError:
        print(f'Cannot load file {str(path)}')
        return None
    return meas, temperature_blackbody


def get_blackbody_temperature_from_path(path: (Path, str)):
    data = Path(path).stem
    if '-' in data:
        data = data.split('-')[0]
    data = data.split('_')
    temperature_blackbody = int(data[-1])
    return path, temperature_blackbody


def load_filter_from_pickle(path_with_wl: tuple):
    path, filter_wavelength = path_with_wl
    meas, temperature_blackbody = load_pickle(path)
    return meas['measurements'][filter_wavelength.value], temperature_blackbody, meas['camera_params']


def get_measurements(path_to_files: (Path, str), filter_wavelength: FilterWavelength, *, omit_ops: Iterable = None):
    paths = Path(path_to_files)
    if paths.is_dir():
        paths = list(path_to_files.glob('*.pkl'))
    elif paths.is_file():
        paths = [paths]
    else:
        raise RuntimeError(f'No .pkl files were found in {path_to_files}.')

    # remove regression models from list (if any exist):
    regression_idx = [i for i, path in enumerate(paths) if "regression_model" in str(path)]
    for idx in regression_idx:
        paths.pop(idx)

    # multithreading loading
    with Pool(8) as pool:
        list_meas = list(tqdm(pool.imap(func=load_filter_from_pickle,
                                        iterable=zip(paths, repeat(filter_wavelength))),
                              total=len(paths), desc="Load measurements"))
    list_meas = list(filter(lambda x: x is not None, list_meas))

    # list into dict
    dict_measurements = {}
    while list_meas:
        meas, t_bb, camera_params = list_meas.pop()
        dict_measurements.setdefault(t_bb, (meas, camera_params))

    # remove invalid operating-points:
    if omit_ops is None:
        omit_ops = []
    for op in omit_ops:
        dict_measurements.pop(op)

    # the dict keys are sorted each time, to save memory space
    list_blackbody_temperatures = list(sorted(dict_measurements.keys()))
    list_power = [calc_rx_power(temperature=t_bb)
                  for t_bb in list_blackbody_temperatures]
    frames = np.stack([dict_measurements[k][0].pop('frames')
                      for k in list_blackbody_temperatures])
    fpa = np.stack([dict_measurements[k][0].pop('fpa')
                   for k in list_blackbody_temperatures])
    housing = np.stack([dict_measurements[k][0].pop('housing')
                       for k in list_blackbody_temperatures])
    cam_params = [p[-1] for p in dict_measurements.values()]
    return frames.squeeze(), fpa.squeeze(), housing.squeeze(), list_power, list_blackbody_temperatures, \
        cam_params


def save_ndarray_as_base64(image: np.ndarray):
    fmt_header = 'data:image/jpeg;base64,'
    image -= image.min()
    image = image - image.max()
    image *= 255
    image = image.astype('uint8')
    with io.BytesIO() as buff:
        Image.fromarray(image).save(buff, format='jpeg')
        return fmt_header + base64.b64encode(buff.getvalue()).decode()


def make_jupyter_markdown_figure(image: np.ndarray, path: (str, Path), title: str = ''):
    """
    Saves a .txt file with the full image encoded as base64.
    The caption should be copied as a whole to a markdown cell in jupyter notebook.

    :param image:
        np.ndarray of dimensions (h, w).
    :param path:
        str of the full path to save the image.
    :param title:
        str. If empty, no caption is inserted to the figure.
    """
    image_base64 = save_ndarray_as_base64(image)
    header = f'<figure><img  style="width:100%" src="{image_base64}">\n'
    if title:
        header += f'<figcaption align = "center">{title}</figcaption>\n'
    header += '</figure>'
    with open(path, 'w') as fp:
        fp.write(header)


def choose_random_pixels(n_pixels: int, img_dims: tuple):
    ndims = len(img_dims)
    return np.random.randint(low=[0] * ndims, high=img_dims, size=(n_pixels, ndims))


def main():
    calc_rx_power(temperature=32)

if __name__ == "__main__":
    main()
