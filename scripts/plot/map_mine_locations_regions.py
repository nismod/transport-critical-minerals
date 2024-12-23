import geopandas as gpd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

country_list = np.array(read_csv('country_list.csv')['name'])

southern_africa = world[np.in1d(world['name'],country_list)]
africa = world[world['continent']=='Africa']

reference_minerals = np.array(["copper","cobalt","manganese","lithium","graphite","nickel"])
reference_mineral_colors = np.array(["#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd"])
Nc = reference_minerals.size

# Manual plot limits becuase life is to short to try to reliably automate this
xl = [8,51]
dxl = abs(np.diff(xl))[0]
yl = [-41,6]
dyl = abs(np.diff(yl))[0]
al = 0.8
marker_size_max = 600
figwidth = 14
w = 0.03
dt = 0.05

tonnage_max = 3680000

fig=plt.figure(figsize=(figwidth,figwidth/2.5/dxl*dyl/(1-dt)*(2.5+2*w)/2.5))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace=w)
for scenario,title,panel,colspan,layer in [['country','2040 National',1,2,'2040_mid_min_threshold_metal_tons'],['region','2040 Regional',3,2,'2040_mid_max_threshold_metal_tons'],['key','',0,1,'']]:

    ax = plt.subplot2grid([1,5],[0,panel],1,colspan=colspan)
    ax.set_ylim(yl)
    ax.spines[['top','right','bottom','left']].set_visible(False)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title,fontsize=16,weight='bold')
    if scenario == 'key':
        ax.set_xlim(xl[0]+0.5*dyl,xl[1])
        tonnage_key = 10**np.arange(1,np.ceil(np.log10(tonnage_max)),1)[::-1]
        Nk = tonnage_key.size
        xk = xl[0] + 0.65*dxl
        dyk = 0.02*dyl
        yk = yl[0] + np.linspace(0.15*dyl,0.4*dyl,Nk)
        xt = xk-0.04*dxl
        for key in ['tonnage','mineral']:    
            yt = yk[-1]+np.diff(yk[-3:-1])
            if key == 'tonnage':
                size_key = marker_size_max*(tonnage_key/tonnage_max)**0.5
                key = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
                key.geometry.plot(ax=ax,markersize=size_key,color='k')
                ax.text(xt,yt,'Mine annual output (tonnes)',weight='bold',va='center')
                for k in range(Nk):
                    ax.text(xk,yk[k],'     {:,.0f}'.format(tonnage_key[k]),va='center')
            else:
                ax.text(xt,yt,'Mineral produced',weight='bold',va='center')
                for k in range(Nk):
                    ax.text(xk,yk[k],'   '+reference_minerals[k].capitalize(),va='center')
                    ax.plot(xk,yk[k],'s',mfc=reference_mineral_colors[k],mec='lightgrey',ms=10)
            Nk = reference_minerals.size
            yk = yk + 0.5*dyl
    else:
        ax.set_xlim(xl)
        mine_sites_df = gpd.read_file('node_locations_for_energy_conversion_'+scenario+'_unconstrained.gpkg',layer=layer)
        mine_sites_df = mine_sites_df[~mine_sites_df["mode"].isin(["mine","city"])]

        africa.plot(ax=ax, color='whitesmoke', edgecolor='white')
        southern_africa.plot(ax=ax, color='lightgrey', edgecolor='white')
        southern_africa.apply(lambda x: ax.annotate(text=x['iso_a3'], xy=x.geometry.centroid.coords[0], ha='center',color='grey',fontsize=10), axis=1)
        total_tons = 0
        for i in range(Nc):
            tons = mine_sites_df[[c for c in mine_sites_df.columns if reference_minerals[i]+'_final' in c]].sum(axis=1)
            marker_size = marker_size_max*(tons/tonnage_max)**0.5
            mine_sites_df.geometry.plot(ax=ax, color=reference_mineral_colors[i], edgecolor='none',markersize=marker_size,alpha=0.7)
            total_tons += np.sum(tons)
        plt.text(xl[0]+0.5*dxl,yl[0]+0.05*dyl,'Total = {:.1f} million tonnes'.format(total_tons/1e6),fontsize=16,weight='bold',ha='center')
        ax.set_facecolor('lightsteelblue')    
   
outfile = 'mine_locations_regions.png'
plt.savefig(outfile,dpi=600)
print('Created '+outfile)
plt.close(fig)
plt.close()