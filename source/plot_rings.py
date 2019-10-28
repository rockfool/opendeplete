import utilities
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib

n_rings = 20
n_radial = 32

cells = [x for x in range(10000,10000 + n_rings * n_radial,1)]
nuclides = ["Xe-135"]

print(cells)

x1,y1 = utilities.get_atoms("/home/cjosey/code/opendeplete/source/test_halfmonth", cells, nuclides)

y = np.zeros(n_rings * n_radial)

for yv in y1:
    y[yv - 10000] = y1[yv]["Xe-135"][3]

# Calculate radii

r_fuel = 1.0
v_fuel = math.pi * r_fuel**2
v_ring = v_fuel / n_rings

# Calculate pin discretization radii
r_rings = np.zeros(n_rings)

# Remaining rings
for i in range(n_rings):
    r_rings[i] = math.sqrt(1.0/(math.pi) * v_ring * (i+1))
print(r_rings)

# Calculate planes
#~ plt.plot(y)
#~ plt.show()

codes = [Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.MOVETO, Path.CLOSEPOLY]

y = y / (v_ring) * n_radial / (0.4096**2)

minv = min(y)
maxv = max(y)
cmap = matplotlib.cm.get_cmap('plasma')
norm = matplotlib.colors.Normalize(vmin=min(y), vmax=max(y))

yn = (y - minv)/(maxv - minv)

fig = plt.figure()
ax = fig.add_axes([0.05, 0.05, 0.7, 0.9])
ind = 0
for i in range(n_rings):
    theta1 = 0
    for j in range(n_radial):
        # Get coordinates
        theta2 = theta1 + 2 * math.pi / n_radial
        print(math.sin(theta1), math.cos(theta1))
        if i > 0:
            x1 = r_rings[i-1] * math.sin(theta1)
            y1 = r_rings[i-1] * math.cos(theta1)
            
            x2 = r_rings[i-1] * math.sin(theta2)
            y2 = r_rings[i-1] * math.cos(theta2)
        else:
            x1 = 0
            y1 = 0
            x2 = 0
            y2 = 0
        x3 = r_rings[i] * math.sin(theta2)
        y3 = r_rings[i] * math.cos(theta2)
        x4 = r_rings[i] * math.sin(theta1)
        y4 = r_rings[i] * math.cos(theta1)

        path = Path([(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x1,y1)])
        print(path)
        patch = patches.PathPatch(path, facecolor=cmap(yn[ind]), lw=1)
        ax.add_patch(patch)
        theta1 += 2 * math.pi / n_radial
        ind += 1
plt.xlim([-1,1])
plt.ylim([-1,1])
ax2 = fig.add_axes([0.8, 0.1, 0.1, 0.8])

cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm)
cb1.set_label('Gd-157 (atoms/cm^3)')
plt.show()
