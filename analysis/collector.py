import argparse
import sys
import threading as th
from functools import partial
from itertools import repeat
from pathlib import Path

sys.path.append(str(Path().cwd().parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

from devices.BlackBodyCtrl import BlackBody
from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheel import FilterWheel


def th_saver(t_bb: int, filter_name: int, images: list, fpa: list, housing: list, path: Path):
    path = path / f'blackbody_temperature_{t_bb:d}_wavelength_{filter_name:d}'
    df = pd.DataFrame(columns=['FPA temperature', 'Housing temperature', 'Filter wavelength nm'],
                      data=[(f,h,wl) for f,h,wl in zip(fpa, housing, repeat(filter_name))])
    df.to_csv(path_or_buf=str(path.with_suffix('.csv')))
    np.save(str(path.with_suffix('.npy')), np.stack(images))


def collect(params: dict, path_to_save: (str, Path), bb_stops: int,
            n_filters: int, n_images: int, bb_max: int, bb_min: int):
    list_t_bb = np.linspace(start=bb_min, stop=bb_max, num=bb_stops, dtype=int)
    if not 0<n_filters<=6:
        raise ValueError(f"n_filters must be 0<n_filters<=6. Received {n_filters}.")
    list_filters = [0, 8000, 9000, 10000, 11000, 12000]   # whats set on the filterwheel
    list_filters = list_filters[:n_filters]
    print(f'BlackBody temperatures: {list_t_bb}C')
    print(f'Filters: {list_filters}nm')
    thread = partial(th.Thread, target=th_saver, daemon=False)
    list_threads = []
    # blackbody = BlackBody()
    filterwheel = FilterWheel()
    camera = CameraCtrl(camera_parameters=params)
    path_to_save = Path(path_to_save)
    if not path_to_save.is_dir():
        path_to_save.mkdir(parents=True)

    dict_images = {}
    length_total = len(list_t_bb) * len(list_filters)
    idx = 1
    for t_bb in list_t_bb:
        # blackbody.temperature = t_bb
        for position, filter_name in enumerate(sorted(list_filters), start=1):
            filterwheel.position = position
            dict_images.setdefault(t_bb, {}).setdefault(filter_name, [])
            list_fpa, list_housing = [], []
            """ do FFC before every filter. This is done under the assumption that the FPA temperature vary 
            significantly during each BB stop. 
            e.g, 3000 image at 60Hz with 6 filters -> 50sec * 6 filters -> ~6 minutes"""
            while not camera.ffc():
                continue
            with tqdm(total=n_images) as progressbar:
                progressbar.set_description_str(f'Filter {filter_name}nm')
                progressbar.set_postfix_str(f'\tBlackBody {t_bb}C\t\tMeasurements: {idx}|{length_total}')
                while len(dict_images[t_bb]) != n_images:
                    dict_images[t_bb].append(camera.image)
                    list_fpa.append(camera.fpa)
                    list_housing.append(camera.fpa)
                    progressbar.update()
            list_threads.append(thread(kwargs=dict(t_bb=t_bb, filter_name=filter_name, path=path_to_save,
                                                   images=dict_images[t_bb].pop(filter_name), fpa=list_fpa.copy(),
                                                   housing=list_housing.copy())))
            list_threads[-1].start()
            idx += 1
    try:
        camera.__del__()
    except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
        pass
    try:
        blackbody.__del__()
    except (ValueError, TypeError, AttributeError, RuntimeError, NameError, KeyError, AssertionError):
        pass
    [p.join() for p in list_threads]


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
parser.add_argument('--blackbody_stops', help=f"How many BlackBody temperatures will be "
                                              f"measured between blackbody_max to blackbody_min.",
                    type=int, default=11)
parser.add_argument('--blackbody_max', help=f"Maximal temperature of BlackBody in C.",
                    type=int, default=70)
parser.add_argument('--blackbody_min', help=f"Minimal temperature of BlackBody in C.",
                    type=int, default=20)
args = parser.parse_args()

params_default = dict(
    ffc_mode='external',  # FFC only when instructed
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

collect(params=params_default, path_to_save=Path(args.path), bb_stops=args.blackbody_stops,
        n_filters=args.n_filters, n_images=args.n_images, bb_max=args.blackbody_max,
        bb_min=args.blackbody_min)
