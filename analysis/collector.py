import argparse
import sys
import threading as th
from functools import partial
from pathlib import Path

sys.path.append(str(Path().cwd().parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from devices.BlackBodyCtrl import BlackBody
from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheel import FilterWheel

BB_LOW = 20
BB_HIGH = 70


def th_saver(t_bb_temperature: int, dict_of_arrays: dict, dict_of_fpa: dict, path: Path):
    keys = list(sorted(dict_of_arrays.keys()))
    values = np.stack([(dict_of_fpa[k], k) for k in keys])
    df = pd.DataFrame(columns=['FPA temperature', 'Filter wavelength nm'], data=values)
    path = path / f'blackbody_temperature_{t_bb_temperature:d}'
    df.to_csv(path_or_buf=str(path.with_suffix('.csv')))
    np.save(str(path.with_suffix('.npy')), np.stack([dict_of_arrays[k] for k in keys]))


def collect(params: dict, path_to_save: (str, Path), bb_stops: int,
            list_filters: (list, tuple), n_images: int):
    list_t_bb = np.linspace(start=BB_LOW, stop=BB_HIGH, num=bb_stops, dtype=int)
    list_filters = list(list_filters) if not isinstance(list_filters, list) else list_filters
    list_filters = [int(p) for p in list_filters]
    print(f'BlackBody temperatures: {list_t_bb}C')
    print(f'Filters: {list_filters}nm')
    thread = partial(th.Thread, target=th_saver, daemon=False)
    list_threads = []
    blackbody = BlackBody()
    filterwheel = FilterWheel()
    camera = CameraCtrl(camera_parameters=params)
    camera.start()
    path_to_save = Path(path_to_save)
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)

    dict_images = {}
    length_total = len(list_t_bb) * len(list_filters)
    idx = 1
    for t_bb in list_t_bb:
        blackbody.temperature = t_bb
        dict_fpa = {}
        for position, filter_name in enumerate(sorted(list_filters), start=1):
            filterwheel.position = position
            """ do FFC before every filter. This is done under the assumption that the FPA temperature vary 
            significantly during each BB stop -> 3000 image at 60Hz with 6 filters -> 50sec * 6 filters -> ~6 minutes"""
            camera.ffc()
            list_images, fpa = [], 0
            with tqdm(total=n_images) as progressbar:
                progressbar.set_description_str(f'Filter {filter_name}nm')
                progressbar.set_postfix_str(f'\tBlackBody {t_bb}C\t\tMeasurements: {idx}|{length_total}')
                while len(list_images) != n_images:
                    list_images.append(camera.image)
                    fpa += camera.fpa
                    progressbar.update()
            dict_fpa[filter_name] = round(fpa / (100 * n_images), 1)  # fpa is in metric 100*C
            dict_images.setdefault(t_bb, {}).setdefault(filter_name, np.stack(list_images))
            idx += 1
        list_threads.append(thread(args=(t_bb, dict_images.pop(t_bb, {}), dict_fpa.copy(), path_to_save,)))
        list_threads[-1].start()
    [p.join() for p in list_threads]
    camera.__del__()
    blackbody.__del__()


parser = argparse.ArgumentParser(description='Measures the distortion in the Tau2 with the Filters.'
                                             'For each BlackBody temperature, images are taken and saved.'
                                             'The images are saved in an np.ndarray '
                                             'with dimensions [n_images, filters, h, w].')
parser.add_argument('--filter_wavelength_list', help="The central wavelength of the Band-Pass filter on the camera",
                    default=[0, 8000, 9000, 10000, 11000, 12000], type=list)
parser.add_argument('--folder_to_save', help="The folder to save the results. Create folder if invalid.",
                    default='measurements')
parser.add_argument('--n_images', help="The number of images to grab.", default=3000, type=int)
parser.add_argument('--blackbody_stops', help=f"How many BlackBody temperatures will be "
                                              f"measured between {BB_LOW}C to {BB_HIGH}C.",
                    type=int, default=11)
args = parser.parse_args()

params_default = dict(
    ffc_mode='manual',  # FFC only when instructed
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

path_default = Path(args.folder_to_save)
col = partial(collect, bb_stops=args.blackbody_stops,
              list_filters=args.filter_wavelength_list, n_images=args.n_images)
col(params=params_default, path_to_save=path_default)
