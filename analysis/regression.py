from os import stat
from typing import Iterable
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import plotly.express as px
import pickle as pkl
import matplotlib.pyplot as plt
from tools import choose_random_pixels


def poly_regress(x: np.ndarray, y: np.ndarray, ord: int) -> np.ndarray:
    """performs a multi-threaded linear regression and returns a list where the ith element are the regression coefficients for the ith feature.

        Parameters:
            x: a 2d matrix where the ith column holds the independent variables for the ith dependent feature
            y: a 2d matrix where the ith column holds the ith dependent feature
            deg: the degree of the fitted polynomial
    """
    map_args = [(x_row, y_row, ord) for x_row, y_row in zip(x, y)]
    with mp.Pool(mp.cpu_count()) as pool:
        regress_res = pool.starmap(np.polyfit, tqdm(map_args, desc="Performing Regression"))

    return np.array(list(regress_res))


class GlRegressor:
    def __init__(self, x_var: Iterable, grey_levels: np.ndarray):
        self.gl_shape = grey_levels.shape
        self.x = np.array(x_var)
        self.y = grey_levels
        self.coeffs = None  # the coefficients of the fitted regression model

    def _reshape_regress(self, var, target_shape=None, phase="pre"):
        """Reshapes the variable before or after the regression fit"""

        if phase =="pre":
            src_shape_rolled = np.roll(range(var.ndim), 2)
            var_trans = var.transpose(src_shape_rolled)
            var_reshaped = var_trans.reshape(np.prod(var_trans.shape[:2]), -1)
        elif phase == "post":
            if target_shape is None:
                res_dim = np.prod(var.shape) // np.prod(self.y.shape[-2:])
                target_shape = (res_dim, *self.y.shape[-2:])
            var_trans = var.T
            var_reshaped = var_trans.reshape(target_shape)
        else:
            raise Exception("Not a valid phase!")
        return var_reshaped

    def get_vars_reshaped(self):
        """Returns a 2d re-shaped version of the variables to facilitate the fitting process, such that each row corresponds to measurements of a different pixel"""
        # broadcast x to the shape of y 
        x_broad = np.broadcast_to(self.x[None, ...], np.roll(self.gl_shape, -1)) 

        # transpose x to be compatible with y
        x_trans = np.transpose(x_broad, np.roll(range(self.y.ndim), 1))
        
        x_reshaped, y_reshaped = self._reshape_regress(x_trans), self._reshape_regress(self.y)

        return x_reshaped, y_reshaped

    def fit(self, deg: int = 1, is_inverse: bool =False, debug: bool = False):
        """performs a pixel-wise polynomial regression for grey_levels vs an independent variable

            Parameters:
                deg: the degree of the fitted polynomial
                is_inverse: flag for calculating the inverse regression (temperature vs grey-levels)
                debug: flag for plotting the regression maps.
        """

        # reshape data to facilitate and parallelize the regression
        x, y = self.get_vars_reshaped()

        if is_inverse:
            x, y = y, x

        regress_res_list = poly_regress(x, y, deg)  # perform the regression

        # reorder results in matrix format
        n_coeffs = deg + 1
        regress_res_mat = regress_res_list.reshape(
            *self.gl_shape[2:], n_coeffs).transpose(2, 0, 1)
        self.coeffs = regress_res_mat

        if debug:
            self.show_coeffs()

    def _assert_coeffs(self):
        """Check if the coefficients are available"""
        assert self.coeffs is not None

    def predict(self, x_query):
        """Predict the target values by applying the regression model on the query point.  

            Parameters:
                x_query: the query points

            Returns: 
                results: len(x_query) feature maps, where results[i] corresponds to the predicted features for x_query[i] 
        """
        if not isinstance(x_query, np.ndarray):
            x_query = np.array(x_query)
        self._assert_coeffs()
        coeffs_for_pred = self._reshape_regress(self.coeffs)

        if x_query.ndim < 2 or not x_query.shape[-2:] == self.gl_shape[-2:]:  # input shape is incompatible with spatial dimensions
            x_row_vec = x_query.flatten()[None, ...]
            x_for_pred = np.repeat(x_row_vec, np.prod(self.gl_shape[-2:]), axis=0)
        else:
            x_for_pred = self._reshape_regress(x_query)

        map_args = [(coeffs, query) for coeffs, query in zip(coeffs_for_pred, x_for_pred)]
        with mp.Pool(mp.cpu_count()) as pool:
            preds = pool.starmap(np.polyval, tqdm(map_args, desc="Predicting"))

        preds = np.array(preds)

        return self._reshape_regress(preds, phase="post")


    def show_coeffs(self):
        """Show a map of the regression model's coefficients"""
        self._assert_coeffs()
        for k, coeffs in enumerate(self.coeffs):
            fig = px.imshow(
                coeffs,
                color_continuous_scale="gray",
                title=f"$p_{k}$ coefficient Map",
            )
            fig.show()


    def plot_rand_pix(self, xlabel=None):

        self._assert_coeffs()
        deg = len(self.coeffs) - 1
        i, j = choose_random_pixels(1, self.gl_shape[2:])
        y = self.y[..., i, j].flatten()
        x = np.repeat(self.x, len(y) // len(self.x))
        p = self.coeffs[:, i, j].flatten()
        y_hat = np.polyval(p, x)
        err = y-y_hat
        rmse = np.sqrt((err ** 2).mean())

        # get fitted model formula:
        regression_parts = [rf"{p[k]:.3f} \times T^{deg-k} +" for k in range(len(p)) if k < deg]
        regression_parts.append(f"{p[-1]:.3f}")
        regression_formula = f"$GL = {' '.join(regression_parts)}$".replace(
            "T^1", "T")

        # plot results:
        _, ax = plt.subplots(figsize=(16, 9))
        ax.scatter(x, y, label="samples")
        x_lim = ax.axes.get_xlim()
        ax.plot(x_lim, np.polyval(p, x_lim), linestyle="--",
                label=f"fitted model (rmse={rmse:.3f})")
        ax.set_title(regression_formula)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("GL")
        ax.grid()
        ax.legend()
        

    def get_coeffs(self):
        """slice coefficient matrix by coefficients and put in a dictionary"""
        self._assert_coeffs()
        coeffs_dict = {
            f"p{k}": self.coeffs[k] for k in range(len(self.coeffs))}

        return coeffs_dict

    def save_model(self, target_path: Path):
        np.save(target_path, self.coeffs)

    def load_model(self, source_path: Path):
        self.coeffs = np.load(source_path)


def main():
    from tools import get_measurements, FilterWavelength
    path_to_files = Path.cwd() / "analysis" / "rawData" / "calib" / "tlinear_0"
    meas_panchromatic, _, _, _, list_blackbody_temperatures, _ =\
        get_measurements(
            path_to_files, filter_wavelength=FilterWavelength.PAN, n_meas_to_load=3)

    gl_regressor = GlRegressor(list_blackbody_temperatures, meas_panchromatic)

    # whether to re-run the regression or take the results from the previous regression:
    re_run_regression = False

    if re_run_regression:
        gl_regressor.fit(is_inverse=False)
        gl_regressor.save_model(
            Path.cwd() / "analysis" / "models" / "t2gl_lin.npy")
    else:
        gl_regressor.load_model(
            Path.cwd() / "analysis" / "models" / "t2gl_lin.npy")
    gl_regressor.plot_rand_pix()
    result = gl_regressor.predict(meas_panchromatic.mean(axis=1))

if __name__ == "__main__":
    main()
