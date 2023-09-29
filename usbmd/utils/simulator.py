""" Lightweight ultrasound simulator that can be used for testing purposes.

- **Author(s)**     : Ben Luijten
- **Date**          : Tue January 31 2023
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from usbmd.utils.pixelgrid import cartesian_pixel_grid


class UltrasoundSimulator:
    """A lightweight ultrasound simulator tool, intended for testing purposes."""

    def __init__(
        self,
        probe=None,
        scan=None,
        ele_pos=None,
        batch_size=1,
        fc=6.25e6,
        c=1540,
        N_scatterers=(20, 40),
    ):
        """Initialize ultrasound simulator

        Args:
            probe (Probe, optional): Class containing probe parameters
                Providing a probe class overrides all other acquisition paremters.
                Defaults to None.
            scan (Scan, optional): Scan object containing parameters and
                beamforming grid. Defaults to None.
            ele_pos (ndarray, optional): Array geometry. Defaults to None.
            batch_size (int, optional): Number of batches. Defaults to 1.
            fc (float, optional): Center frequency. Defaults to 6.25e6.
            c (int, optional): Speed-of-Sound. Defaults to 1540.
            N_scatterers (tuple, optional): [min, max] number of point scatterers.
                Will be used when no specific points are provided.
                Defaults to [20, 40].
        """

        # Set acquisition parameters
        if scan and probe:
            print("Probe and scanclass recognized, ignoring manual parameters")
            self.fc = scan.fc
            self.c = scan.c
            self.ele_pos = probe.ele_pos

        else:
            self.fc = fc
            self.c = c

            if ele_pos:
                self.ele_pos = ele_pos
            else:
                self.ele_pos = np.stack(
                    [
                        np.linspace(-19.0e-3, 19.0e-3, 128).T,
                        np.zeros((128,)),
                        np.zeros((128,)),
                    ],
                    axis=1,
                )

        self.batch_size = batch_size
        self.wvln = scan.wvln
        self.ele_pos = self.ele_pos[:, 0] - np.min(self.ele_pos[0, 0])

        # Set grid
        if scan is not None:
            self.grid = scan.grid
        else:
            self.grid = cartesian_pixel_grid(
                [-19e-3, 19e-3], [0, 63e-3], dx=self.wvln / 8, dz=self.wvln / 8
            )

        self.dx = self.grid[0, 1, 0] - self.grid[0, 0, 0]
        self.dz = self.grid[1, 0, 2] - self.grid[0, 0, 2]

        self.min_scatterers = N_scatterers[0]
        self.max_scatterers = N_scatterers[1]

        self.Nx = self.grid.shape[1]
        self.Nz = self.grid.shape[0]

        # Simulation parameters
        self.Nt = 2 * (self.Nz) * self.dz / c
        self.dt = (1 / fc) / 4  # fs = 4*fc
        self.t = self.dt * np.arange(0, np.round(self.Nt / self.dt))
        self.x = self.dx * np.arange(0, self.Nx)
        self.z = self.dz * np.arange(0, self.Nz)

    def pulse(self, tau):
        """Transmit pulse"""
        sig = 2e-7
        return np.exp(-0.5 * ((self.t - tau) / sig) ** 2) * np.sin(
            2 * np.pi * self.fc * (self.t - tau)
        )

    def loc(self, x0, z0):
        """TODO: add docstring"""
        sig_x = 1 * self.dx
        xg, zg = np.meshgrid(self.x, self.z)
        return np.exp(-0.5 * (((xg - x0) / sig_x) ** 2 + ((zg - z0) / sig_x) ** 2))

    def generate(self, points=None):
        """generates pairs of input/target RF data"""

        inp = []
        tar = []
        for i in range(self.batch_size):
            # Get positions of point scatterers
            if isinstance(points, list):
                scatterers = len(points[i])
                points_x = points[i, 0]
                points_z = points[i, 1]
            else:
                if isinstance(points, int):
                    scatterers = points
                else:
                    scatterers = np.random.randint(
                        self.min_scatterers, self.max_scatterers
                    )
                points_x = self.dx * self.Nx * np.random.rand(scatterers)
                points_z = self.dx * self.Nz * np.random.rand(scatterers)

            # Calculate response
            s_i = 0
            y_i = 0

            for j in range(scatterers):
                d_trans = points_z[j] / self.c
                tau_j = d_trans + np.sqrt(
                    ((points_x[j] - self.ele_pos) / self.c) ** 2
                    + (points_z[j] / self.c) ** 2
                )
                s_i = s_i + np.array(
                    [self.pulse(tau_j[k]) for k in range(0, len(tau_j))]
                )
                y_i = y_i + self.loc(points_x[j], points_z[j])
            inp.append(s_i.T)
            tar.append(y_i)

        return np.array(inp), np.array(tar)


if __name__ == "__main__":
    sim = UltrasoundSimulator()
    data = sim.generate()

    img1 = abs(data[0].squeeze())
    img2 = data[1].squeeze()
    img2 = cv2.blur(img2, (20, 20))
    res_img1 = cv2.resize(img1, dsize=img2.shape[::-1])

    fig, axs = plt.subplots(1, 3)

    aspect_ratio = (data[1].shape[1] / data[1].shape[2]) / (
        data[0].shape[1] / data[0].shape[2]
    )
    axs[0].imshow(abs(img1), aspect=aspect_ratio)
    axs[1].imshow(img2)
    axs[2].imshow(img2)
    axs[2].imshow(res_img1 / res_img1.max() + img2 / img2.max())
    plt.show()
