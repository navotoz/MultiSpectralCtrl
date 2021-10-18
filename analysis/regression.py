from typing import Iterable
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import plotly.express as px
import pickle as pkl

from analysis.tools import choose_random_pixels


def poly_regress(x: np.ndarray, y: np.ndarray, ord: int, is_inverse:bool=False) -> np.ndarray:
    # TODO: 1. simplify predict to allow easy and simple predictions (only polyval). 2. add a function to allow for parallel prediction (e.g with full matrix size) 3. add R^2 to the rand_pix_plot method
    """performs a multi-threaded linear regression and returns a list where the ith element are the regression coefficients for the ith feature.

        Parameters:
            x: a 2d matrix where the ith column holds the independent variables for the ith dependent feature
            y: a 2d matrix where the ith column holds the ith dependent feature
            deg: the degree of the fitted polynomial
    """
    # Parallel regression
    if is_inverse:
        x, y = y, x
        map_args = [(temperature, y, ord) for temperature in x]
    else:
        map_args = [(x, temperature, ord) for temperature in y]

    with mp.Pool(mp.cpu_count()) as pool:
        regress_res = pool.starmap(np.polyfit, tqdm(map_args, desc="Performing Regression"))

    return np.array(list(regress_res))


class GlRegressor:
    def __init__(self, x_var: Iterable, grey_levels: np.ndarray, is_inverse: bool = False):
        self.gl_shape = grey_levels.shape
        self.x = np.array(x_var)
        self.y = grey_levels
        self.coeffs = None  # the coefficients of the fitted regression model
        self._is_inverse = is_inverse

    @property
    def is_inverse(self):
        return self._is_inverse
    
    @is_inverse.setter
    def is_inverse(self, is_inverse):
        if is_inverse != self._is_inverse:
            self._is_inverse = is_inverse
            self.coeffs = None  # clear coeffs to avoid confusion

    def fit(self, deg: int = 1, debug: bool = False):
        """performs a pixel-wise polynomial regression for grey_levels vs an independent variable

            Parameters:
                x_var: an iterable holding the independent variables for the regression model
                grey_levels: the measured grey-levels, assuming the 1st dimension corresponds to the samples, and the others are spatial dimensions
                deg: the degree of the fitted polynomial
                is_inverse: flag for calculating the inverse regression (temperature vs grey-levels)
                debug: flag for plotting the regression maps.
        """

        # reshape data to facilitate and parallelize the regression
        x = self.x.repeat(self.gl_shape[1])
        y = self.y.reshape(-1, self.gl_shape[2] * self.gl_shape[3]).T

        regress_res_list = poly_regress(x, y, deg, self.is_inverse)  # perform the regression

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

        self._assert_coeffs()
        coeffs_for_val = self.coeffs.transpose(1, 2, 0).reshape(-1, len(self.coeffs))
        if len(x_query.shape) > 1: # it's a full grey-level measurement matrix -  need to reshape and run by frames:
            x_query_reshaped = x_query.transpose(2, 3, 0, 1).reshape(
                self.gl_shape[2] * self.gl_shape[3], -1)  # TODO: make transposition more general
            map_args = [(coeffs, query) for coeffs, query in zip(coeffs_for_val, x_query_reshaped)]
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.starmap(np.polyval, tqdm(map_args, desc="Predicting"))
            results = np.array(results)
            return results.T.reshape(x_query.shape)
        else:
            results = np.apply_along_axis(np.polyval, 1, coeffs_for_val, x_query)
            return results.T.reshape(len(x_query), *self.gl_shape[2:])


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


    # def plot_rand_pix(self, xlabel=None):
    #     self._assert_coeffs()
    #     deg = len(self.coeffs) - 1
    #     i, j = choose_random_pixels(1, self.gl_shape[2:])

    #     res = self.predict()
    #     coeffs = np.polyfit(x, y, deg)
    #     plt.figure(figsize=(16, 9))
    #     sns.regplot(x=x, y=y, order=deg)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.title(title)
    #     plt.grid()

    #     n_coeffs = deg+1
    #     regression_parts = [
    #         rf"{coeffs[k]:.3f} \times T^{deg-k} +" for k in range(n_coeffs) if k < deg]
    #     regression_parts.append(f"{coeffs[-1]:.3f}")
    #     regression_formula = f"$GL = {' '.join(regression_parts)}$".replace(
    #         "T^1", "T")
    #     # TODO: finalize regression formula in display
    #     display(Latex(
    #         fr"Regression result: {regression_formula}"))
    #     return coeffs
    #         res = plot_regression(x, y, deg, xlabel, ylabel="Grey-Level",
    #                             title=f"Regression model for random pixel {(i[0], j[0])}")
    #     return res

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
        gl_regressor.fit(is_inverse=False, debug=True)
        gl_regressor.save_model(Path.cwd() / "analysis" / "models" / "gl2t_lin.npy")
    else:
        gl_regressor.load_model(
            Path.cwd() / "analysis" / "models" / "gl2t_lin.npy")

    result = gl_regressor.predict(list_blackbody_temperatures)

if __name__ == "__main__":
    main()
