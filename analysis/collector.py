import argparse
import sys
import threading as th
from datetime import datetime
from functools import partial
from itertools import repeat
from pathlib import Path
from time import sleep

import yaml

sys.path.append(str(Path().cwd().parent))

from utils.misc import save_ndarray

import numpy as np
import pandas as pd
from tqdm import tqdm

from devices.BlackBodyCtrl import BlackBody
from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheel import FilterWheel


def th_saver(t_bb_: int, filter_name_: int, images: list, fpa: list, housing: list, path: Path,
             save_gif: bool = False):
    path = path / f'blackbody_temperature_{t_bb_:d}_wavelength_{filter_name_:d}'
    df = pd.DataFrame(columns=['FPA temperature', 'Housing temperature', 'Filter wavelength nm'],
                      data=[(f, h, wl) for f, h, wl in zip(fpa, housing, repeat(filter_name_))])
    df.to_csv(path_or_buf=str(path.with_suffix('.csv')))
    np.save(str(path.with_suffix('.npy')), np.stack(images))
    if save_gif:
        save_ndarray(np.stack(images), dest_folder=path.parent, type_of_files='gif', name=path.stem)


parser = argparse.ArgumentParser(description='Measures the distortion in the Tau2 with the Filters.'
                                             'For each BlackBody temperature, images are taken and saved.'
                                             'The images are saved in an np.ndarray '
                                             'with dimensions [n_images, filters, h, w].')
parser.add_argument('--n_filters', help="How many filters on the filterwheel to use."
                                        "The central wavelength of the "
                                        "Band-Pass filter is [0, 8000, 9000, 10000, 11000, 12000]nm",
                    type=int, default=6)
parser.add_argument('--path', help="The folder to save the results. Create folder if invalid.",
                    default='measurements')
parser.add_argument('--n_images', help="The number of images to grab.", default=3000, type=int)
parser.add_argument('--ffc_period', help=f"The number of frames between automatic FFCs. Works only for auto mode."
                                         f"The camera calculates the frames as 30Hz. "
                                         f"e.g, setting to 1800 frames means a 60 seconds between FFC.", type=int,
                    default=60 * 30)  # FFC every 1 minutes - 30Hz * 60 seconds (although the camera actually runs 60Hz)
parser.add_argument('--tlinear', help=f"The grey levels are linear to the temperature as: 0.04 * t - 273.15.",
                    action='store_true')
parser.add_argument('--blackbody_stops', help=f"How many BlackBody temperatures will be "
                                              f"measured between blackbody_max to blackbody_min.",
                    type=int, default=11)
parser.add_argument('--blackbody_max', help=f"Maximal temperature of BlackBody in C.",
                    type=int, default=70)
parser.add_argument('--blackbody_min', help=f"Minimal temperature of BlackBody in C.",
                    type=int, default=20)
parser.add_argument('--gif', help=f"Saves a gif of each measurement.", action='store_true')
args = parser.parse_args()

params = dict(
    ffc_mode='auto',  # FFC every ffc_period
    ffc_period=int(args.ffc_period),
    isotherm=0x0000,
    dde=0x0000,
    tlinear=int(args.tlinear),  # T-Linear disabled. The scene will not represent temperatures, because of the filters.
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

if __name__ == "__main__":
    path_to_save = Path(args.path) / datetime.now().strftime("%Y%m%d_h%Hm%Ms%S")
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)
    with open(str(path_to_save / 'params.yaml'), 'w') as fp:
        yaml.safe_dump(params, stream=fp, default_flow_style=False)
    n_filters, n_images = args.n_filters, args.n_images
    list_t_bb = np.linspace(start=args.blackbody_min, stop=args.blackbody_max, num=args.blackbody_stops, dtype=int)
    if not 0 < n_filters <= 6:
        raise ValueError(f"n_filters must be 0<n_filters<=6. Received {n_filters}.")
    list_filters = [0, 8000, 9000, 10000, 11000, 12000]  # whats set on the filterwheel
    list_filters = list_filters[:n_filters]
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
        for position, filter_name in enumerate(sorted(list_filters), start=1):
            filterwheel.position = position
            dict_images.setdefault(t_bb, {}).setdefault(filter_name, [])
            list_fpa, list_housing = [], []
            """ do FFC before every filter. This is done under the assumption that the FPA temperature vary
            significantly during each BB stop.
            e.g, 3000 image at 60Hz with 6 filters -> 50sec * 6 filters -> ~6 minutes"""
            while not camera.ffc():
                continue
            sleep(1)  # clears the buffer after the FFC
            with tqdm(total=n_images) as progressbar:
                progressbar.set_description_str(f'Filter {filter_name}nm')
                progressbar.set_postfix_str(f'\tBlackBody {t_bb}C\t\tMeasurements: {idx}|{length_total}')
                while len(dict_images[t_bb][filter_name]) != n_images:
                    dict_images[t_bb][filter_name].append(camera.image)
                    list_fpa.append(camera.fpa)
                    list_housing.append(camera.housing)
                    progressbar.update()
            list_threads.append(thread(kwargs=dict(t_bb_=t_bb, filter_name_=filter_name, path=path_to_save,
                                                   images=dict_images[t_bb].pop(filter_name), fpa=list_fpa.copy(),
                                                   housing=list_housing.copy(), save_gif=args.gif)))
            list_threads[-1].start()
            idx += 1
    try:
        camera.terminate()
    except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
        pass
    try:
        blackbody.__del__()
    except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
        pass
    [p.join() for p in list_threads]
