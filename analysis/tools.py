import base64
import io
import os
import pickle
import warnings
from enum import Enum
from itertools import repeat
from multiprocessing.dummy import Pool
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from tqdm import tqdm
import multiprocessing as mp
import random


class FilterWavelength(Enum):
    """An enumeration for Flir's LWIR filters, where the keys and values are the central frequencies of the filters in nano-meter."""
    PAN = 0
    nm8000 = 8000
    nm9000 = 9000
    nm10000 = 10000
    nm11000 = 11000
    nm12000 = 12000


def c2k(T): return np.asarray(T) + 273.15
def k2c(T): return np.asarray(T) - 273.15


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

    # %%
    # setup

    # constants:
    KB = 1.380649e-23  # [Joule/Kelvin]
    H = 6.62607004e-34  # [m^2*kg/sec]
    C = 2.99792458e8  # [m/sec]

    # define grid of wavelengths over the entire band of interest with a 1nm
    # granularity:
    wl_band = (7_000, 14_000)  # in [nm]
    d_lambda = 1e-9
    wl_grid = np.arange(start=wl_band[0], stop=wl_band[-1], dtype=int)

    # %%
    # Auxiliary functions

    def bb(lamda: float, T: float):
        """calculates spectral radiance (Plank's function)

        Parameters: lamda: the wavelength in meters T: the temperature in Kelvin
        """
        return 2 * H * C ** 2 / np.power(lamda, 5) * \
            1 / (np.exp(H * C / (KB * T * lamda)) - 1)

    def read_trans_resp(sheet_name):
        """Read the transmittance response from an excel sheet into a pandas
        series"""
        trans_resp_raw = pd.read_excel(Path(os.path.dirname(__file__), "FiltersResponse.xlsx"),
                                       sheet_name=sheet_name, engine='openpyxl', index_col=0, usecols=[0, 1]).dropna().squeeze()
        # convert Transmission from percents to normalized values:
        trans_resp_raw /= 100
        trans_resp_raw.index *= 1e3  # convert index to [nm]

        # interpolate the values over the wavelength's grid:
        interp_model = interp1d(x=trans_resp_raw.index,
                                y=trans_resp_raw.values, kind="linear")

        trans_resp = interp_model(wl_grid)
        return trans_resp

    # %%
    # calculate optical elements responses. microbolometer response was taken
    # from the paper "Spectral response of microbolometers for hyperspectral
    # imaging"
    # https://www.researchgate.net/publication/316709375_Spectral_response_of_microbolometers_for_hyperspectral_imaging
    sens_resp = read_trans_resp("microbolometer")
    lens_resp = read_trans_resp("lens_680138-002")

    if filt == FilterWavelength.PAN:
        # assume ideal rectangular filter with 3000nm bandwidth and a constant
        # amplification of 1
        filt_resp = np.ones_like(wl_grid)
        filt_grid = wl_grid
    else:
        filt_resp = read_trans_resp(str(filt.value))

        # remove entries Bw far away from the central frequency:
        bw = 1000
        below_bw = wl_grid < (filt.value - bw)
        above_bw = wl_grid > (filt.value + bw)
        inside_bw = np.bitwise_not(np.bitwise_or(below_bw, above_bw))
        filt_grid = wl_grid[inside_bw]
        filt_resp = filt_resp[inside_bw]
        # convert Transmission from percent to normalized values:

    # %%
    # calculate the equivalent system's transmittance to black-body radiation:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if isinstance(temperature, np.ndarray):  # temperature is a vector - need to vectorize all inputs:
            plank_trunc = bb(filt_grid[..., None] * d_lambda, c2k(temperature[None, ...]))
        else:
            plank_trunc = bb(filt_grid * d_lambda, c2k(temperature))

    overlap_idx = [i for i, wl in enumerate(wl_grid) if wl in filt_grid]
    trans_resp = sens_resp[overlap_idx] * lens_resp[overlap_idx] * filt_resp
    plank_resp = plank_trunc.T * trans_resp

    # %%
    # Plot results:
    def rad_power(density, dl): return np.nansum(density * dl, axis=0)
    if debug:
        import matplotlib.pyplot as plt

        # validate integral over whole spectrum:
        sigma = 5.670373e-8  # [W/(m^2K^4)]

        if isinstance(temperature, np.ndarray):
            temperature_eg = temperature[0]
            plank_resp_eg =  plank_resp[0]
        else:
            temperature_eg = temperature
            plank_resp_eg = plank_resp


        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            plank = bb(np.arange(1e6, dtype=int) *
                       d_lambda, c2k(temperature_eg))
        L = rad_power(plank, d_lambda)  # integrate over plank
        T_hat = k2c(np.power(L / sigma * np.pi, 1 / 4))
        print(f"Input temperature: {temperature_eg:.4f}")
        print(
            f"Estimated temperature (by integrating over plank's function): {T_hat:.4f}")

        fig, ax = plt.subplots(1, 2, figsize=(16, 9))

        # plot transmittance response:
        ax[0].plot(wl_grid, sens_resp * 100,
                   label="Microbolometer (sensor) transmittance")
        ax[0].plot(wl_grid, lens_resp * 100, label="Lense transmittance")
        if filt != FilterWavelength.PAN:
            ax[0].plot(filt_grid, filt_resp * 100,
                       label=f"{filt.value//1000} $\mu m$ filter transmittance")
        ax[0].plot(filt_grid, trans_resp * 100, label=f"combined transmittance")
        ax[0].set_title("Optical Transmittance")
        ax[0].set_ylabel("Transmittance [%]")
        ax[0].set_xlabel("$\lambda$[nm]")
        ax[0].grid()
        ax[0].legend()

        # plot raw spectrum and transmitted spectrum:
        ax[1].plot(plank, label="complete spectrum")
        ax[1].plot(wl_grid, plank[wl_grid], label="TAU-2 bandwidth")
        ax[1].plot(filt_grid, plank_resp_eg, label="transmitted spectrum")
        ax[1].set_ylabel("Spectral Radiance")
        ax[1].set_xlabel("$\lambda$[nm]")
        ax[1].grid()
        ax[1].legend()
        ax[1].set_title("The filtered segment of the spectral radiance")
        ax[1].set_xlim([0, 2e4])
        if filt == FilterWavelength.PAN:
            fig.suptitle(f"Pan-Chromatic at T={temperature_eg}[C]")
        else:
            fig.suptitle(f"{filt.value} nm Filter at T={temperature_eg}[C]")
        plt.show()

    return rad_power(plank_resp.T, d_lambda)


def prefilt_cam_meas(cam_meas: np.ndarray, *, first_valid_meas: int = 0, med_filt_sz: int = 2):
    """pre-filtering of raw measurements taken using Flir's TAU-2 LWIR camera.
    The filtering pipe is based on insights gained and explored in the
    'meas_inspection' notebook available under the same parent directory.

        Parameters: 
            cam_meas: a 4D hypercube that contains the data collected by a set of measurements. 
            first_valid_idx: the first valid measurement in all operating points. All measurements taken before that will be discarded. 
            med_filt_sz: the size of the median filter used to clean dead pixels from the measurements.
    """

    cam_meas_valid = cam_meas[:, first_valid_meas:, ...]
    meas_shape = cam_meas_valid.shape
    med_filt_shape = np.pad(np.array((med_filt_sz, med_filt_sz)), pad_width=(
        len(meas_shape)-3, 0), constant_values=1)
    map_args = [(meas, tuple(med_filt_shape)) for meas in cam_meas_valid]
    with mp.Pool(mp.cpu_count()) as pool:
        cam_meas_filt = pool.starmap(median_filter, tqdm(
            map_args, desc="Pre-Filtering Measurements"))

    return np.array(cam_meas_filt)


def load_pickle(path: Path):
    try:
        path, temperature_blackbody = get_blackbody_temperature_from_path(path)
        with open(str(path), 'rb') as fp:
            meas = pickle.load(fp)
    except ValueError:
        print(f'Cannot load file {str(path)}')
        return None
    return meas, temperature_blackbody


def get_blackbody_temperature_from_path(path: Union[Path, str]):
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


def load_bb_dict(path_to_files, filter_wavelength, n_meas_to_load):
    paths = Path(path_to_files)
    if paths.is_dir():
        paths = list(path_to_files.glob('*.pkl'))
    elif paths.is_file():
        paths = [paths]
    else:
        raise RuntimeError(f'No .pkl files were found in {path_to_files}.')

    # remove non-related pkl files from list (if any exist):
    remove_paths = [
        path for path in paths if "blackbody_temperature" not in str(path)]
    for path in remove_paths:
        paths.remove(path)

    # load a limited number of operating-points (useful for debug):
    if n_meas_to_load is not None:
        meas_idx_to_load = np.random.randint(0, len(paths), n_meas_to_load)
        new_list = []
        for op in range(len(paths)):
            if op in meas_idx_to_load:
                new_list.append(paths[op])
        paths = new_list

    # multithreading loading
    with Pool(mp.cpu_count()) as pool:
        list_meas = list(tqdm(pool.imap(func=load_filter_from_pickle,
                                        iterable=zip(paths, repeat(filter_wavelength))),
                              total=len(paths), desc="Load measurements"))
    list_meas = list(filter(lambda x: x is not None, list_meas))

    # list into dict
    dict_measurements = {}
    while list_meas:
        meas, t_bb, camera_params = list_meas.pop()
        dict_measurements.setdefault(t_bb, (meas, camera_params))
    return dict_measurements


def _load_filter_fast(path):
    return path.stem, np.load(path)


def get_measurements(path_to_files: Union[Path, str], filter_wavelength: FilterWavelength, *, n_meas_to_load: int = None, fast_load: bool = False, do_prefilt=False, temperature_units="C"):
    if not fast_load:
        dict_measurements = load_bb_dict(
            path_to_files, filter_wavelength, n_meas_to_load)

        # the dict keys are sorted each time, to save memory space
        list_blackbody_temperatures = list(sorted(dict_measurements.keys()))
        frames = np.stack([dict_measurements[k][0].pop('frames')
                           for k in list_blackbody_temperatures])
        fpa = np.stack([dict_measurements[k][0].pop('fpa')
                        for k in list_blackbody_temperatures])
        housing = np.stack([dict_measurements[k][0].pop('housing')
                            for k in list_blackbody_temperatures])
        cam_params = [p[-1] for p in dict_measurements.values()]

    else:
        path_to_filt = path_to_files / filter_wavelength.name
        f_list = [file for file in path_to_filt.glob("*") if file.is_file()]
        if n_meas_to_load is not None:
            f_list = random.sample(f_list, k=n_meas_to_load)
        # with Pool(mp.cpu_count()) as pool: res = pool.imap(_load_filter_fast,
        #     tqdm(f_list, desc="Load measurements"))
        res = []
        for file in tqdm(f_list, desc="Load measurements"):
            res.append(_load_filter_fast(file))
        list_blackbody_temperatures = [int(tup[0]) for tup in res]
        frames = np.stack([tup[1] for tup in res])
        fpa = housing = cam_params = np.array([])

    list_power = [calc_rx_power(temperature=t_bb, filt=filter_wavelength)
                  for t_bb in tqdm(list_blackbody_temperatures, desc="calculating power")]

    if temperature_units == "K":
        list_blackbody_temperatures = [c2k(t)
                                       for t in list_blackbody_temperatures]

    if do_prefilt:
        frames = prefilt_cam_meas(frames)

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


def make_jupyter_markdown_figure(image: np.ndarray, path: Union[str, Path], title: str = ''):
    """
    Saves a .txt file with the full image encoded as base64. The caption should
    be copied as a whole to a markdown cell in jupyter notebook.

    :param image: np.ndarray of dimensions (h, w). :param path: str of the full
        path to save the image. :param title: str. If empty, no caption is
        inserted to the figure.
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
    idx_mat = np.random.randint(
        low=[0] * ndims, high=img_dims, size=(n_pixels, ndims))

    # organize indices in lists of lists:
    idx_list = [list(col) for col in idx_mat.T]
    return idx_list


def calc_r2(y_hat, y):
    """calculates the coefficient of determination of a model fit"""
    y_bar = y.mean()
    ss_res = ((y-y_hat) ** 2).sum()
    ss_tot = ((y-y_bar) ** 2).sum()
    r_2 = 1 - ss_res / ss_tot

    return r_2


def split_by_filt(path_to_files, filter_wavelength: FilterWavelength, *, n_meas_to_load: int = None):
    """split the raw black-body measurement files to subfiles by filter"""
    for filt_wl in FilterWavelength:
        target_path = path_to_files / filt_wl.name
        if not target_path.is_dir():
            target_path.mkdir()

        dict_measurements = load_bb_dict(
            path_to_files, filt_wl, n_meas_to_load)
        for temperature in dict_measurements.keys():
            frames = np.array(dict_measurements[temperature][0]["frames"])
            np.save(target_path / f"{temperature}.npy", frames)


def find_parent_dir(parent_dir_name):
    cur_path = Path.cwd()
    while not (cur_path / parent_dir_name).is_dir():
        cur_path = cur_path.parent
    return (cur_path / parent_dir_name)


def main():
    # calc_rx_power(32, FilterWavelength.nm9000, debug=True)
    arr = np.arange(10)
    res_vec = calc_rx_power(arr, FilterWavelength.nm9000, debug=True)

if __name__ == "__main__":
    main()
