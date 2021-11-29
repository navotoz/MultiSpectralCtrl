import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
from collections.abc import Iterable
import sys

sys.path.append('analysis')
from tools import FilterWavelength, get_measurements, calc_r2


class Gaussian:
    def __init__(self) -> None:
        self.amp = self.mean = self.std = self.bias = None
        self.ndim = None

    @staticmethod
    def _gaussian_1d(x, amp, mu, sigma, bias) -> np.ndarray:
        return amp * np.exp(-0.5*(x - mu)**2 / sigma**2) + bias

    @staticmethod
    def _gaussian_2d(grid, amp, mu_x, mu_y, sigma_x, sigma_y, bias):

        x_cent = grid[0] - mu_x
        y_cent = grid[1] - mu_y
        return amp * np.exp(-0.5*((x_cent / sigma_x)**2 + (y_cent / sigma_y)**2)) + bias

    @staticmethod
    def _res_grid(grid, ndim):
        return np.asarray(grid).reshape(ndim, -1)

    def fit(self, grid, f_grid, p0) -> np.ndarray:
        """Fit a shifted gaussian based on the provided samples and initial parameters guess. returns the covariance matrix of the estimated parameters"""

        # Currently supports only 2D gaussian
        if isinstance(grid, Iterable) or (isinstance(grid, np.ndarray) and grid[0].ndim > 1):
            self.ndim = 2
            grid_resh = self._res_grid(grid, self.ndim)
            f_grid_resh = f_grid.ravel()
            p0_resh = [p0[0], *p0[1], *p0[2], p0[3]]
            best_vals, cov = curve_fit(
                self._gaussian_2d, grid_resh, f_grid_resh, p0=p0_resh)
            self.amp = best_vals[0]
            self.mean = best_vals[1:3]
            self.std = best_vals[3:5]
            self.bias = best_vals[-1]
        else:
            self.ndim = 1
            grid_resh = grid
            f_grid_resh = f_grid
            p0_resh = p0
            best_vals, cov = curve_fit(
                self._gaussian_1d, grid_resh, f_grid_resh, p0=p0_resh)
            self.amp, self.mean, self.std, self.bias = best_vals
        return cov

    def predict(self, x):
        if self.ndim == 2:
            res = self._gaussian_2d(self._res_grid(
                x, self.ndim), self.amp, *self.mean, *self.std, self.bias)
            return res.reshape(x[0].shape)
        else:
            res = self._gaussian_1d(
                x, self.amp, self.mean, self.std, self.bias)
            return res

    def eval_fit(self, y_hat, y_meas, grid=None, debug=True):
        if grid is None:
            x, y = np.meshgrid(
                np.arange(y_hat.shape[1]), np.arange(y_hat.shape[0]))
        else:
            x, y = grid

        if debug:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(x, y, y_meas, label="Measurements")
            ax.plot_surface(x, y, y_hat, label="Fitted Model", alpha=0.7)
            ax.set_title("Gaussian Model Fitting (Blue=Data, Red=Model Fit)")
            plt.show()
        return calc_r2(y_hat, y_meas)

    @staticmethod
    def _eg():
        """This method is used as an example for using the class"""
        path_to_files = Path("analysis/rawData") / 'calib' / 'tlinear_0'
        meas_panchromatic, _, _, _, _, _ =\
            get_measurements(path_to_files, filter_wavelength=FilterWavelength.PAN,
                             fast_load=True, do_prefilt=False, n_meas_to_load=1)

        meas_to_fit = meas_panchromatic.mean(axis=0)
        x, y = np.meshgrid(
            np.arange(meas_to_fit.shape[1]), np.arange(meas_to_fit.shape[0]))

        bias = (meas_to_fit[0, 0] + meas_to_fit[0, -1] +
                meas_to_fit[-1, 0] + meas_to_fit[-1, -1]) / 4
        amp = -np.abs(meas_to_fit.min() - bias)
        cen = np.asarray(meas_to_fit.shape) // 2
        sigma = cen

        init_vals = [amp, cen, sigma, bias]

        gauss_model = Gaussian()
        cov = gauss_model.fit((x, y), meas_to_fit, init_vals)
        gauss_hat = gauss_model.predict((x, y))
        r2 = gauss_model.eval_fit(gauss_hat, meas_to_fit, debug=True)
        print(f"Model Fit R^2 is {r2:.3f}")


def main():

    gauss_model = Gaussian()
    gauss_model._eg()


if __name__ == "__main__":
    main()
