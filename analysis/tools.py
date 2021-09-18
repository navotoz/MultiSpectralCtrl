import threading as th
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from devices.BlackBodyCtrl import BlackBody
from devices.Camera.CameraProcess import CameraCtrl
from devices.FilterWheel.FilterWheel import FilterWheel


def th_saver(t_bb_temperature: int, dict_of_arrays: dict, dict_of_fpa: dict, path: Path):
    keys = list(sorted(dict_of_arrays.keys()))
    values = np.stack([(dict_of_fpa[k], k) for k in keys])
    df = pd.DataFrame(columns=['FPA temperature', 'Filter wavelength nm'], data=values)
    path = path / f'blackbody_temperature_{t_bb_temperature:d}'
    df.to_csv(path_or_buf=str(path.with_suffix('.csv')))
    np.save(str(path.with_suffix('.npy')), np.stack([dict_of_arrays[k] for k in keys]))


def collect(params: dict, path_to_save: (str, Path), list_t_bb: (list, tuple),
            list_filters: (list, tuple), n_images: int):
    thread = partial(th.Thread, target=th_saver, daemon=False)
    list_threads = []
    blackbody = BlackBody()
    filterwheel = FilterWheel()
    camera = CameraCtrl(camera_parameters=params)
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
            dict_fpa[filter_name] = fpa / n_images
            dict_images.setdefault(t_bb, {}).setdefault(filter_name, np.stack(list_images))
            idx += 1
        list_threads.append(thread(args=(t_bb, dict_images.pop(t_bb, {}), dict_fpa.copy(), path_to_save,)))
        list_threads[-1].start()
    [p.join() for p in list_threads]
