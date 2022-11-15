from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm
import torch
import math


def generate_spheres(npoints):
    harmonic_cfs = [[0, 0], [1, 1], [0, 1], [-1, 1], [2, 2], [1, 2], [0,2], [-1, 2], [-2, 2]]

    phi = np.linspace(0, np.pi, npoints)
    theta = np.linspace(0, 2*np.pi, npoints)
    phi, theta = np.meshgrid(phi, theta)
    spheres = torch.empty((9, npoints, npoints))

    for i, [m, l] in enumerate(harmonic_cfs):
        spheres[i, :, :] = torch.tensor(sph_harm(m, l, theta, phi).real)
    return spheres

def generate_sphere_coords(npoints):
    phi = np.linspace(0, np.pi, npoints)
    theta = np.linspace(0, 2*np.pi, npoints)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

@dataclass
class HarmonicSpheres:
    spheres = generate_spheres(100)
    x, y, z = generate_sphere_coords(100)

@dataclass
class Plenoxel:
    npoints = 100

    center : torch.Tensor
    alpha = 1

    harmonics = HarmonicSpheres()

    def __post_init__(self):
        self.weights = torch.rand((3, 9))

    def get_total(self):
        spheres = self.harmonics.spheres
        weights = self.weights

        res = torch.ones((self.npoints, self.npoints, 4))
        res[:, :, :3] = (weights @ spheres.view((9, self.npoints**2))).T.view(self.npoints, self.npoints, 3)
        res = res.clamp(min=0, max=1)
        return res

    def get_cartesian(self):
        return self.harmonics.x, self.harmonics.y, self.harmonics.z

    # Compare sphere
    def __sub__(self, other):

        npoints = self.npoints
        myfft = torch.fft.fft(self.get_total().view(npoints**2, 4)[:, :3], dim = 1)
        otherfft = torch.fft.fft(other.get_total().view(100*100, 4)[:, :3], dim = 1)

        return abs((myfft - otherfft).mean(dim=(0, 1)))

@dataclass
class PlenoxelGrid:
    res: int

    def __post_init__(self):
        self.plenoxels = np.empty((self.res, self.res, self.res), dtype=object)
        for x in range(self.res):
            for y in range(self.res):
                for z in range(self.res):
                    center = torch.Tensor([x,y,z])
                    self.plenoxels[x][y][z] = Plenoxel(center)
        #self.plenoxels = np.array([Plenoxel(x,y,z) for x in range(self.res) for y in range(self.res) for z in range(self.res)]).reshape(self.res, self.res, self.res)

    def interpolate(x, y, z):
        # Interpolate weights to nearest surrounding plenoxels.
        print("Not implemented yet!")

def plot_spheres(spheres : list[Plenoxel], titles: list[str] = []):
    if len(titles) == 0:
        for i, sphere in enumerate(spheres):
            rounded = torch.round(sphere.center, decimals=2)
            titles.append(f"{rounded.numpy()}")
    x,y,z = spheres[0].get_cartesian()

    total = len(spheres)
    cols = math.floor(math.sqrt(total))
    rows = math.ceil(total / cols)

    fig = plt.figure(figsize=(10,10))

    for index in range(0, total):
        sphere = spheres[index]
        title = titles[index]
        ax = fig.add_subplot(rows, cols, index+1, projection='3d')
        ax.set_title(title)
        ax.plot_surface(x, y, z,  rstride=1, cstride=1, facecolors=sphere.get_total().numpy(), shade=False)
        # Turn off the axis planes
        ax.set_axis_off()
    plt.show()