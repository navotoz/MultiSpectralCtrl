import argparse

from analysis.tools import collect

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

collect(params=camera_parameters, path_to_save=args.folder_to_save, list_t_bb=args.blackbody_temperatures_list,
        list_filters=args.filter_wavelength_list, n_images=args.n_images)