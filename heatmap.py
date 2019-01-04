import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('temp/sims_2000','rb') as f:
    sims = pickle.load(f)

with open('temp/force_2000','rb') as f:
    force = pickle.load(f)

keys1 = set(sims.keys())
keys1 = set([frozenset(k) for k in keys1])
keys2 = set(force.keys())
keys = keys1 & keys2

print(len(keys))

x = []
y = []
for key in keys:
    y.append(force[key])
    key = tuple(key)
    try:
        x.append(sims[key])
    except:
        x.append(sims[(key[1],key[0])])

x = np.asarray(x)
y = np.asarray(y)

print(x.shape,y.shape)

heatmap,xedges,yedges = np.histogram2d(x,y,bins=50)
extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]

print(heatmap)

plt.clf()
plt.imshow(heatmap.T,extent=extent,origin='lower')

plt.savefig('heatmap_2000_')

