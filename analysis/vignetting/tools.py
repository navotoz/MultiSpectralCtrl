import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from symfit import parameters, Fit, Model, variables, Poly


def radius(h: int, w: int) -> np.ndarray:
    mesh = np.meshgrid(np.linspace(-0.5, 0.5, w), np.linspace(-0.5, 0.5, h))
    radii = np.sqrt(mesh[0] ** 2 + mesh[1] ** 2)
    radii -= radii.min()
    return radii


def fit_surface(order: int, radii: np.ndarray, real: np.ndarray, norm_type: (str, None) = None,
                verbose: bool = True):
    p_str, z_max, z_min = f'', 1, 0
    for i in range(order):
        p_str += f'c{i} '
    params = parameters(p_str)
    p_dict = {}
    for index in range(order):
        p_dict[index] = params[index]
    r, z = variables('r, z')
    model = Model({z: Poly(p_dict, r).as_expr()})
    print(model) if verbose else None

    z_ = real.copy()
    if norm_type and norm_type == 'minmax':
        z_min = z_.min()
        z_ -= z_min
        z_max = z_.max()
        z_ /= z_max
    elif norm_type and norm_type == 'constant':
        z_ /= 2 ** 14
    fit = Fit(model, r=radii.astype('float64'), z=z_.astype('float64'))
    fit_result = fit.execute()
    est = model(r=radii, **fit_result.params).z
    if norm_type and norm_type == 'minmax':
        est *= z_max
        est += z_min
    elif norm_type and norm_type == 'constant':
        est *= 2 ** 14
    print(fit_result) if verbose else None
    return est, fit_result.params


def calcRxPower(temperature: float, central_wl: int, *, is_ideal_filt: bool = False, debug=False):
    """Calculates the power emmited by the the black body, according to Plank's law of black-body radiation, 
    after being filtered by the applied narrow-banded spectral filter.
    
        Parameters:
            temperature - the temperatureerature of the black body [C]
            central_wl - the central wavelength of the filter [nm]
            is_ideal_filt - set to True to use an optimal rectangular filter centered about the central wavelength. 
            Otherwise, use the practical filter according to Thorlab's characterization.
    """
    # # constants:
    KB = 1.380649e-23  # [Joule/Kelvin]
    H = 6.62607004e-34  # [m^2*kg/sec]
    C = 2.99792458e8  # [m/sec]

    # spectral radiance (Plank's function)
    bb = lambda lamda, T: 2 * H * C ** 2 / np.power(lamda, 5) * 1 / (np.exp(H * C / (KB * T * lamda)) - 1)

    # Celsuis to Kelvin:
    c2k = lambda T: T + 273.15
    k2c = lambda T: T - 273.15

    bw = 1000  # [nm]
    if is_ideal_filt:  # assume ideal rectangular filter with 1000nm bandwidth and a constant amplification of 1
        d_lambda = 1e-9
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
    plank_unfilt = bb(lambda_grid, c2k(temperature))
    plank_filt = plank_unfilt * bp_filter

    if debug:
        import matplotlib.pyplot as plt
        base_grid = np.arange(2e4) * 1e-9  # [m]
        fig, ax = plt.subplots(1, 2)

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
            ax[1].plot(base_grid * 1e9, bb(base_grid, c2k(temperature)), label="complete spectrum")

        ax[1].plot(lambda_grid * 1e9, plank_unfilt, label="segment of interest")
        ax[1].plot(lambda_grid * 1e9, plank_filt, label="filtered spectrum")
        ax[1].set_ylabel("Spectral Radiance")
        ax[1].set_xlabel("$\lambda$[nm]")
        ax[1].grid()
        ax[1].legend()
        ax[1].set_title("The filtered segment of the spectral radiance")

        plt.show(block=False)

        # validate integral over whole spectrum:
        valid_grid = np.arange(1e6) * 1e-9  # [m]
        sigma = 5.670373e-8  # [W/(m^2K^4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            L = np.nansum(bb(valid_grid, c2k(temperature)) * 1e-9)
        T_hat = k2c(np.power(L / sigma * np.pi, 1 / 4))
        print(f"Input temperature: {temperature:.4f}")
        print(f"Estimated temperature (by integrating over plank's function): {T_hat:.4f}")

    return (plank_filt * d_lambda).sum()
