import argparse
import logging
import threading as th
from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis.undistort.Blackbody.BlackBodyCtrl import BlackBody
from devices.Camera.Tau.Tau2Grabber import Tau2Grabber
from devices.FilterWheel.FilterWheel import FilterWheel


def th_saver(name_of_filter: int, dict_of_arrays: dict):
    for t_bb_name, arr in dict_of_arrays.items():
        name = path_to_save / f'wavelength_{name_of_filter:d}_blackbody_{t_bb_name:d}.npy'
        np.save(str(name), arr)


def loader(path):
    return path.stem, np.load(path)


parser = argparse.ArgumentParser(description='Measures the distortion in the Tau2 with the Filters.'
                                             'For each BlackBody temperature, images are taken and saved.'
                                             'The images are saved in an np.ndarray with dimensions [n_images, h, w].')
parser.add_argument('--filter_wavelength_list', help="The central wavelength of the Band-Pass filter on the camera",
                    default=[0, 8000, 9000, 10000, 11000, 12000], type=list)
parser.add_argument('--folder_to_save', help="The folder to save the results. Create folder if invalid.",
                    default='data')
parser.add_argument('--n_images', help="The number of images to grab.", default=2000, type=int)
parser.add_argument('--blackbody_temperatures_list', help="The temperatures for the BlackBody.",
                    default=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
args = parser.parse_args()

camera_parameters = dict(
    ffc_mode='auto',  # do FFC as needed.
    isotherm=0x0000,
    dde=0x0000,
    tlinear=0x0000,  # T-Linear disabled. The scene will not represent temperatures, because of the filters.
    gain='high',
    agc='manual',
    ace=0,
    sso=0,
    contrast=0,
    brightness=0,
    brightness_bias=0,
    fps=0x0004,  # 60Hz
    lvds=0x0000,  # disabled
    lvds_depth=0x0000,  # 14bit
    xp=0x0002,  # 14bit w/ 1 discrete
    cmos_depth=0x0000,  # 14bit pre AGC
)

th_list = []
blackbody = BlackBody(logging_level=logging.INFO)
filterwheel = FilterWheel()
camera = Tau2Grabber()
camera.set_params_by_dict(camera_parameters)
path_to_save = Path(args.folder_to_save)
if not path_to_save.is_dir():
    path_to_save.mkdir(parents=True)

t_bb_list = args.blackbody_temperatures_list
filters_list = args.filter_wavelength_list
dict_images = {}
with tqdm(total=len(t_bb_list) * len(filters_list)) as progressbar:
    for filter_name in filters_list:
        filterwheel.position = filter_name
        progressbar.set_postfix_str(f'Filter {filter_name}nm')
        for t_bb in t_bb_list:
            blackbody.temperature = t_bb
            progressbar.set_postfix_str(f'BlackBody {t_bb}C')
            camera.ffc()
            list_images = []
            for i in range(args.n_images):
                list_images.append(camera.grab(to_temperature=False))
            dict_images.setdefault(filter_name, {}).setdefault(t_bb, np.stack(list_images))
            progressbar.update()
        th_list.append(th.Thread(target=th_saver, args=(filter_name, dict_images.get(filter_name, {}).copy(),)))
        th_list[-1].start()
        t_bb_list = list(reversed(t_bb_list))  # to begin the next filter from the closest temperature
[p.join() for p in th_list]
