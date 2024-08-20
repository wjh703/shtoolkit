

import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
import numpy as np

# from tvg_toolkit.sh import Spharm

from shtoolkit.shspecial import sea_level_equation

from shtoolkit.shload import read_load_love_num
from shtoolkit.shtrans import cilm2grid
import cartopy.crs as ccrs
# from cartopy.util import add_cyclic_point
import time

# breakpoint()
def read_file(path):
    with open(path) as f:
        content = []
        for line in f:
            line = line.replace('D', 'E').strip().split()
            content.append(float(line[2]))
        content = np.array(content).reshape(180, 360)
    return content

def read_field(path, lmax):
    cilm = np.zeros((2, lmax+1, lmax+1))
    with open(path) as f:
        for line in f:
            break
        for line in f:
            line = line.replace('D', 'E').strip().split()
            l, m = int(line[1]), int(line[2])
            if m > lmax:
                break
            if l > lmax:
                continue
            cilm[:, l, m] = float(line[3]), float(line[4])
    return cilm

lmax = 60
ant = read_file('D:\\tvg_toolkit\\fingerprint_approach\\测试sle\\ant_001.mas')
ant_geo_ilm = read_field('D:\\tvg_toolkit\\fingerprint_approach\\测试sle\\ant_001_rotation.geo', 60)
ant_upl_ilm = read_field('D:\\tvg_toolkit\\fingerprint_approach\\测试sle\\ant_001_rotation.upl', 60)
ant_geo_norot_ilm = read_field('D:\\tvg_toolkit\\fingerprint_approach\\测试sle\\ant_001_no_rotation.geo', 60)
ant_upl_norot_ilm = read_field('D:\\tvg_toolkit\\fingerprint_approach\\测试sle\\ant_001_no_rotation.upl', 60)
ant_geo = cilm2grid(ant_geo_ilm, 89, lmax)
ant_upl = cilm2grid(ant_upl_ilm, 89, lmax)
ant_geo_norot = cilm2grid(ant_geo_norot_ilm, 89, lmax)
ant_upl_norot = cilm2grid(ant_upl_norot_ilm, 89, lmax)

lat = np.linspace(89.5, -89.5, 180)
lon = np.linspace(0.5, 359.5, 360)

oc = np.loadtxt('D:\\tvg_toolkit\\tvg_toolkit\\data\\oc_func_100km.txt')[:, 2].reshape(180, 360)

lln_file = 'D:\\tvg_toolkit\\tvg_toolkit\\data\\lln_PREM.txt'
lln = read_load_love_num(lln_file, lmax)
_, geo_norot, upl_norot, _ = sea_level_equation(ant, oc, lln, lmax, 'kgm2mass', False)
start = time.time()
_, geo, upl, _ = sea_level_equation(ant, oc, lln, lmax, 'kgm2mass', True)
print(time.time()-start)


lon_x, lat_y = np.meshgrid(lon, lat)
fig = plt.figure(layout='constrained')
# fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, subplot_kw={'projection': ccrs.PlateCarree()})
# gs = GridSpec(2, 2, width_ratios=[15, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree())
ax1.spines['geo'].set_linewidth(0.8)
ax1.set_global()
ax1.coastlines() # type: ignore
p = ax1.pcolormesh(lon_x, lat_y, ant_upl-ant_upl_norot, transform=ccrs.PlateCarree(), cmap='jet')
# fig.colorbar(p, ax=ax, orientation='horizontal', extend='both')
ax2= fig.add_subplot(212, projection=ccrs.PlateCarree())
ax2.spines['geo'].set_linewidth(0.8)
ax2.set_global()
ax2.coastlines() # type: ignore
ax2.pcolormesh(lon_x, lat_y, upl-upl_norot, transform=ccrs.PlateCarree(), cmap='jet', norm=p.norm)

# ax = fig.add_subplot(224)
cbar = fig.colorbar(p, ax=[ax1, ax2], location='right', orientation='vertical', extend='both', shrink=0.6)
plt.show()
