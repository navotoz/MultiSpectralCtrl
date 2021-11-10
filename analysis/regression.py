import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import pickle as pkl
import matplotlib.pyplot as plt
from tools import FilterWavelength, calc_rx_power, find_parent_dir
from tools import choose_random_pixels, calc_r2, c2k, k2c
import multiprocessing as mp
from tqdm import tqdm


class GlRegressor:
    def __init__(self, is_parallel=False, x_label="T"):
        # the physical independent variable (used for plotting)
        self.x_label = x_label
        # flag for using parallel computing when fitting and predicting
        self.is_parallel = is_parallel
        self.coeffs = None  # the coefficients of the fitted regression model
        self.is_inverse = False

    def _decimate_var(self, var, train_val_rat, offset=0):
        dec_rate = int(1 / train_val_rat)
        var_dec = var[offset::dec_rate]
        return var_dec

    def _pre_proc_var(self, var, train_val_rat, n_meas=None):
        """Returns a 2d re-shaped version of the variables to facilitate the fitting process, such that each row corresponds to measurements of a different pixel"""

        var_dec = self._decimate_var(var, train_val_rat)
        # true if this is the independent variable (a 1d vector)
        if n_meas not in var.shape:
            # broadcast x to the shape of y's columns dimension:
            var_reshaped = np.repeat(var_dec, n_meas)
        else:
            # reshape matrix to a 2D matrix where the columns are the raveled spatial pixels and the columns are
            var_reshaped = var_dec.reshape(np.prod(var_dec.shape[:2]), -1)
        return var_reshaped.squeeze()


    def fit(self, x: np.ndarray, y: np.ndarray, deg: int = 1, feature_power: float = 1, train_val_rat: float = 0.5, debug: bool = False):
        """performs a pixel-wise polynomial regression for grey_levels vs an independent variable

            Parameters:
                x: the dependent variable
                y: the grey-level measurements of the camera (axis 0 corresponds to the independent variable vector length)
                deg: the degree of the polynomial model to fit the data
                feature_power: the power of x to be taken as the independent features (phi) for the regression
                train_val_rat: the portion of the data to be used for training the regression
                debug: flag for plotting the regression maps.
        """

        print("Pre-Processing Variables For Regression...")
        self.feature_power = feature_power
        phi_x = x ** feature_power if feature_power != 1 else x

        # pre-process variables for the parallelized regression:
        n_meas = y.shape[1]
        phi_x_regress, y_regress = self._pre_proc_var(
            phi_x, train_val_rat, n_meas), self._pre_proc_var(y, train_val_rat, n_meas)
        print("Pre-Processing is Complete!")
        
        # perform the regression
        print("Performing the regression (this might take a few seconds/minutes, depending on the data size...)")
        if self.is_parallel:
            map_args = [(phi_x_regress, y_row, deg) for y_row in y_regress.T]
            with mp.Pool(mp.cpu_count()) as pool:
                regress_res_list = pool.starmap(np.polyfit, tqdm(
                    map_args, desc="Performing Regression"))
            regress_res = np.array(list(regress_res_list)).T
        else:
            regress_res = np.polyfit(phi_x_regress, y_regress, deg)
        print("Regression is complete!")

        self.coeffs = regress_res.reshape(-1, *y.shape[2:])

        if debug:
            self.plot_rand_pix(phi_x, y, train_val_rat)

    def _assert_coeffs(self):
        """Check if the coefficients are available"""
        assert self.coeffs is not None


    @staticmethod
    def _solve_inverse(coeffs, y):
        if len(coeffs) == 2:  # model is linear
            a, b = coeffs
            sol = (y-b) / a
        elif len(coeffs) == 3:  # model is quadratic
            a, b, c = coeffs
            # we assume that the solution is the positive root:
            sol = (-b + np.sqrt(b ** 2 - 4*a*(c-y))) / (2*a)
        else:
            raise Exception("Model Order is too high to provide a solution")
        return sol

    def _get_preds(self, x, is_inverse, is_pixel_wise):

        
        # reshape coefficients:
        self._assert_coeffs()  
        coeffs_for_pred = self.coeffs.reshape(len(self.coeffs), -1)

        # reshape query points to comply with the required arguments shape:
        query_shape = x.shape
        if len(query_shape) < 2:  # query is a 1d vector -> repeat it to match the number of pixels
            query_for_pred = x
        else:
            query_for_pred = self._pre_proc_var(x, 1, x.shape[0])

        # predict:
        if self.is_parallel:
            if is_inverse:
                map_args = [(coeffs, query)
                            for coeffs, query in zip(coeffs_for_pred.T, query_for_pred.T)]
                pred_func = self._solve_inverse
            else:
                if is_pixel_wise:
                    map_args = [(coeffs, queries)
                                for coeffs, queries in zip(coeffs_for_pred.T, query_for_pred.T)]
                else:
                    map_args = [(coeffs, query_for_pred) for coeffs in coeffs_for_pred.T]
                pred_func = np.polyval
            with mp.Pool(mp.cpu_count()) as pool:
                preds_lst = pool.starmap(pred_func, tqdm(
                    map_args, desc="Predicting"))
            preds = np.asarray(preds_lst).T

        else:
            if is_inverse:
                preds = self._solve_inverse(coeffs_for_pred, query_for_pred)
            else:
                if is_pixel_wise:
                    preds = np.asarray([np.polyval(coeffs, queries) for coeffs, queries in
                                        zip(coeffs_for_pred.T, query_for_pred.T)]).reshape(self.coeffs.shape[-2:])
                else:
                    preds = np.apply_along_axis(
                        np.polyval, 0, coeffs_for_pred, query_for_pred)

        return preds

    def predict(self, query_pts:np.ndarray, is_inverse=False, is_pixel_wise=False):
        """Predict the target values by applying the model to the query points.  

            Parameters:
                query_pts: the query points for the prediction
                is_inverse: flag indicating whether to predict x from y instead of y from x
                is_pixel_wise: indicator for applying the prediction for the queries only in the corresponding pixel (the prediction of each query point corresponds to a single pixel). This requires that the query points have the same dimensions as the spatial dimensions of the coefficients. Relevant only in forward predictions.
            Returns: 
                results: len(x_query) feature maps, where results[i] corresponds to the predicted features for x_query[i] 
        """

        if not isinstance(query_pts, np.ndarray):
            query_pts = np.asarray(query_pts)

        if not is_inverse and self.feature_power != 1:
            query_pts = query_pts ** self.feature_power  # convert to features

        # get predictions:
        preds = self._get_preds(query_pts, is_inverse, is_pixel_wise)

        # reshape predictions:
        preds_reshaped = preds.reshape(-1, *self.coeffs.shape[-2:])
        if query_pts.ndim > 3:  # further reshaping is required
            preds_reshaped = preds_reshaped.reshape(query_pts.shape)

        # convert predicted features back to the original independent variable:
        if is_inverse and self.feature_power != 1:
            preds_reshaped **= 1 / self.feature_power
        print("Predictions are ready!")

        return preds_reshaped

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

    def plot_rand_pix(self, x, y, train_val_rat):

        self._assert_coeffs()
        deg = len(self.coeffs) - 1
        i, j = choose_random_pixels(1, y.shape[2:])

        y_pix = y[..., i, j].squeeze()
        x_train, y_train = self._decimate_var(
            x, train_val_rat, offset=0), self._decimate_var(y_pix, train_val_rat, offset=0)
        x_val, y_val = self._decimate_var(
            x, train_val_rat, offset=1), self._decimate_var(y_pix, train_val_rat, offset=1)

        p = self.coeffs[:, i, j].flatten()
        y_hat = np.polyval(p, x)
        r_2 = calc_r2(y_hat, y_pix.T)

        # get fitted model formula:
        regression_parts = [
            rf"{p[k]:.3e} x^{self.feature_power * (deg-k)} +" for k in range(len(p)) if k < deg]
        regression_parts.append(f"{p[-1]:.3e}")
        regression_formula = f"$GL = {' '.join(regression_parts)}$".replace(
            f"x^1", "x")

        # plot results:
        n_meas = y_pix.shape[1]
        _, ax = plt.subplots()
        ax.scatter(x_train.repeat(n_meas).reshape(y_train.shape),
                   y_train, label="regression samples")
        ax.scatter(x_val.repeat(n_meas).reshape(y_val.shape),
                   y_val, label="validation samples")
        x_lim = ax.axes.get_xlim()
        x_query = np.linspace(start=x_lim[0], stop=x_lim[1], num=100)
        ax.plot(x_query, np.polyval(p, x_query), linestyle="--", color="k",
                label=f"fitted model (R^2={r_2:.4f})")
        ax.set_title(regression_formula)
        ax.grid()
        ax.legend()
        x_label = f"$x^{self.feature_power}$"
        if self.is_inverse:
            ax.set_xlabel("GL")
            ax.set_ylabel(x_label)

        else:
            ax.set_xlabel(x_label)
            ax.set_ylabel("GL")

        plt.show()

    def eval(self, gt, estimate, op_pts, debug=False, ax_lbls=None):
        """evaluate the performance of the regression over both training and validation sets"""
        if gt.ndim == 4:  # gt is a matrix of intensities measurements - need to first take the average over the samples
            gt = gt.mean(axis=1)
        else:
            estimate = estimate.mean(axis=1)
            gt = gt.repeat(np.prod(estimate.shape[1:])).reshape(estimate.shape)
        est_err = gt - estimate
        err_df = pd.DataFrame(est_err.reshape(
            est_err.shape[0], -1).T, columns=np.round(op_pts, decimals=2))

        if debug:
            _, ax = plt.subplots()
            np.sqrt((err_df**2).mean()).plot.bar(ax=ax)
            ax.set_title("Estimation RMSE")
            if ax_lbls is not None:
                ax.set_xlabel(ax_lbls["xlabel"])
                ax.set_ylabel(ax_lbls["ylabel"])
            else:
                ax.set_xlabel("Operating Point")
                ax.set_ylabel("Estimation Error")
            ax.grid(axis="y")
            plt.show()
            plt.tight_layout()

        return err_df

    def get_coeffs(self):
        """slice coefficient matrix by coefficients and put in a dictionary"""
        self._assert_coeffs()
        coeffs_dict = {
            f"p{k}": self.coeffs[k] for k in range(len(self.coeffs))}

        return coeffs_dict

    def save_model(self, target_path: Path):
        model = {"coeffs": self.coeffs, "is_inverse": self.is_inverse,
                 "feature_power": self.feature_power}
        with open(target_path, "wb") as file:
            pkl.dump(model, file)

    def load_model(self, source_path: Path):
        with open(source_path, "rb") as file:
            model = pkl.load(file)

        self.coeffs = model["coeffs"]
        self.is_inverse = model["is_inverse"]
        self.feature_power = model["feature_power"]


def load_regress_model(path_to_models, model_name):
    f_name = model_name + ".pkl"
    model = GlRegressor()
    model.load_model(path_to_models / f_name)
    return model


def colorization_pipeline(pan_image, pan_model_name="t2gl_4ord", filt=FilterWavelength.nm9000):
    base_path = find_parent_dir("MultiSpectralCtrl") / 'analysis'
    path_to_models = base_path / 'models'

    temperature_to_pan = load_regress_model(path_to_models, pan_model_name)
    power_to_filt = load_regress_model(path_to_models, f"p2gl_{filt.name}")

    temperature_map = k2c(temperature_to_pan.predict(pan_image, is_inverse=True))
    power_filt = calc_rx_power(
        temperature_map.flatten(), filt).reshape(pan_image.shape)
    filt_image = power_to_filt.predict(power_filt, is_pixel_wise=True)
    # map_args = [(temperature, filt) for temperature in temperature_map.flatten()]
    # # with mp.Pool(mp.cpu_count()) as pool:
    # #     power_map = pool.starmap(calc_rx_power, tqdm(
    # #         map_args, desc="Evaluating power"))
    # power_map = [calc_rx_power(temperature, filt) for temperature in tqdm(
    #     temperature_map.flatten(), desc="Evaluating power")]
    # filt_image = power_to_filt.predict(power_map)
    return filt_image



def main():
    # from tools import get_measurements, FilterWavelength
    # path_to_files = Path.cwd() / "analysis" / "rawData" / "calib" / "tlinear_0"
    # meas_panchromatic, _, _, _, list_blackbody_temperatures, _ =\
    #     get_measurements(
    #         path_to_files, filter_wavelength=FilterWavelength.PAN, fast_load=True, n_meas_to_load=3)
    # x = np.asarray(c2k(list_blackbody_temperatures))
    # gl_regressor = GlRegressor(is_parallel=False)
    # # # gl_regressor.fit(x, meas_panchromatic, deg=1,
    # # #                  feature_power=4, train_val_rat=0.5, debug=True)
    # gl_regressor.load_model(Path.cwd() / "analysis" /
    #                         "models" / "t2gl_1ord.pkl")
    # y_hat = gl_regressor.predict(x)
    # ax_lbls = {"xlabel": "Temperature[C]", "ylabel": "Grey-levels"}
    # eval_res = gl_regressor.eval(meas_panchromatic, y_hat,
    #                              list_blackbody_temperatures, debug=True, ax_lbls=ax_lbls)
    # x_hat = k2c(gl_regressor.predict(meas_panchromatic, is_inverse=True))
    # eval_res = gl_regressor.eval(k2c(x), x_hat, list_blackbody_temperatures, debug=True)

    data_base_dir = Path(
        r"C:\Users\omriber\Documents\Thesis\MultiSpectralCtrl\download")
    data_fname = "cnt1_20210830_h15m53s38.npy"
    data = np.load(Path(data_base_dir, data_fname))

    synth = colorization_pipeline(data[0, 0])
    real = data[1, 0]
    a = 1


if __name__ == "__main__":
    main()
