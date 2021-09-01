import argparse
import logging
import threading as th
from pathlib import Path

import numpy as np
from tqdm import tqdm

from analysis.undistort.Blackbody.BlackBodyCtrl import BlackBody
from devices.Camera.Tau.Tau2Grabber import Tau2Grabber
from devices.FilterWheel.FilterWheel import FilterWheel
import pandas as pd


def th_saver(name_of_filter: int, dict_of_arrays: dict):
    keys = sorted(dict_of_arrays.keys())
    df = pd.DataFrame(columns=['BlackBody Temperature C'], data=keys)
    df.to_csv(path_or_buf=path_to_save / f'wavelength_{name_of_filter:d}.csv')
    name = path_to_save / f'wavelength_{name_of_filter:d}.npy'
    np.save(str(name), np.stack([dict_of_arrays[k] for k in keys]))


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

list_threads = []
blackbody = BlackBody(logging_level=logging.INFO)
filterwheel = FilterWheel()
camera = Tau2Grabber()
camera.set_params_by_dict(camera_parameters)
path_to_save = Path(args.folder_to_save)
if not path_to_save.is_dir():
    path_to_save.mkdir(parents=True)

list_t_bb = args.blackbody_temperatures_list
list_filters = args.filter_wavelength_list
dict_images = {}
length_total = len(list_t_bb) * len(list_filters)
idx = 1
for position, filter_name in enumerate(list_filters, start=1):
    filterwheel.position = position
    for t_bb in list_t_bb:
        blackbody.temperature = t_bb
        list_images = []
        with tqdm(total=args.n_images) as progressbar:
            progressbar.set_description_str(f'Filter {filter_name}nm')
            progressbar.set_postfix_str(f'\tBlackBody {t_bb}C\t\tMeasurements: {idx}|{length_total}')
            while len(list_images) != args.n_images:
                image = camera.grab(to_temperature=False)
                if image is not None:
                    list_images.append(image)
                    progressbar.update()
        dict_images.setdefault(filter_name, {}).setdefault(t_bb, np.stack(list_images))
        idx += 1
    list_threads.append(th.Thread(target=th_saver, args=(filter_name, dict_images.pop(filter_name, {}),)))
    list_threads[-1].start()
    list_t_bb = list(reversed(list_t_bb))  # to begin the next filter from the closest temperature
[p.join() for p in list_threads]
