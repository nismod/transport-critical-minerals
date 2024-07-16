import geopandas as gpd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

country_list = np.array(read_csv('country_list.csv')['name'])

southern_africa = world[np.in1d(world['name'],country_list)]

data_pa = gpd.read_file('ProtectedAreasSouthernAfrica/protected_areas_southern_africa.shp')
print(data_pa.shape[0],'protected areas')  
data_low = gpd.read_file('./LastOfWildSouthernAfrica/low_southern_africa.shp')
print(data_low.shape[0],'last of wild')
data_kba = gpd.read_file('./KeyBiodiversityAreasSouthernAfrica/kba_southern_africa.shp')
print(data_kba.shape[0],'key biodiversity areas')

# Manual plot limits becuase life is to short to reliably automate this
xl = [11,51]
yl = [-36,6]
al = 0.5

fig=plt.figure(figsize=(6,6*np.diff(yl)[0]/np.diff(xl)[0]))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
ax = plt.subplot(111)
ax.set_xlim(xl)
ax.set_ylim(yl)
ax.set_frame_on(False)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
southern_africa.plot(ax=ax, color='white', edgecolor='grey',alpha=al)
pmarks = []
for data, color,label in [[data_pa,'tab:blue',"Protected areas"],
                          [data_low,'firebrick',"Last of wild"],
                          [data_kba,'tab:green',"Key biodiversity areas"]]:
    data['geometry'].plot(ax=ax, color=color, alpha=.5,label=label)
    pmarks.append(Patch(facecolor=color, label=label,alpha=al))
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=[*handles,*pmarks],loc=(0.64,0.1))
outfile = 'environmental_areas.png'
plt.savefig(outfile,dpi=300)
print('Created '+outfile)
plt.close(fig)
