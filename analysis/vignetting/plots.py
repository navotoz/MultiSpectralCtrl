import matplotlib.pyplot as plt
import numpy as np


def plot_meas_vs_bb(list_wavelength: list, list_bb: list, measurements: dict):
    h, w = np.random.randint(low=[0, 0], high=measurements[list_wavelength[0]][list_bb[0]].shape, size=2)
    print(f'Random pixel {h, w}')
    for wl in list_wavelength:
        values = [(t, p[h, w]) for t, p in measurements[wl].items()]
        values = list(sorted(values, key=lambda x: x[0]))
        plt.figure()
        plt.scatter([p[0] for p in values], [p[1] for p in values], c='r')
        plt.plot([p[0] for p in values], [p[1] for p in values], c='b')
        plt.xlabel('BlackBody Temperature [C]')
        plt.ylabel('Measured Temperature [C]')
        plt.title(f'Measured vs BlackBody GT for $\lambda=${wl / 1000:.1f}' + '$_{\mu m}$')
        plt.grid()
        plt.show(block=False)
        # plt.close()


def plot_surface_fit(real: np.ndarray, est: np.ndarray):
    fig = plt.figure(dpi=100)
    ax = plt.axes(projection='3d')
    w_mesh, h_mesh = np.meshgrid(np.arange(real.shape[1]), np.arange(real.shape[0]))
    ax.plot_surface(w_mesh, h_mesh, real, cmap='Greens', alpha=0.4)
    ax.plot_surface(w_mesh, h_mesh, est, cmap='Reds')
    fig.tight_layout()
    fig.show()
