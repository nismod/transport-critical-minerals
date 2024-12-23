import geopandas as gpd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

country_list = np.array(read_csv('country_list.csv')['name'])

southern_africa = world[np.in1d(world['name'],country_list)]
africa = world[world['continent']=='Africa']

reference_minerals = np.array(["copper","cobalt","manganese","lithium","graphite","nickel"])
reference_minerals_columns = np.array([m +"_initial_stage_production_tons_0.0_in_country" for m in reference_minerals])
reference_mineral_colors = np.array(["#f46d43","#fdae61","#fee08b","#c2a5cf","#66c2a5","#3288bd"])
Nc = reference_minerals.size

# Manual plot limits becuase life is to short to try to reliably automate this
xl = [8,51]
dxl = abs(np.diff(xl))[0]
yl = [-50,6]
dyl = abs(np.diff(yl))[0]
al = 0.8
w = 0.05
dt = 0.04
figwidth = 16

fig=plt.figure(figsize=(figwidth,figwidth/(3+2*w)/dxl*dyl/(1-dt)))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1-dt,wspace = w)
   
panel = 1
scenario = 'unconstrained'
for layer,key in [['2022_baseline','tonnage'],["2030_mid_min_threshold_metal_tons",'none'],['2040_mid_min_threshold_metal_tons','mineral']]:

    mine_sites_df = gpd.read_file('node_locations_for_energy_conversion_country_'+scenario+'.gpkg',layer=layer)
    mine_sites_df["total_tons"] = mine_sites_df[reference_minerals_columns].sum(axis=1)
    mine_sites_df['color'] = reference_mineral_colors[np.argmax(mine_sites_df[reference_minerals_columns],axis=1)]
    
    tonnage_max = np.max(mine_sites_df["total_tons"])
    marker_size_max = 600
    marker_size = marker_size_max*(mine_sites_df["total_tons"]/tonnage_max)**0.5
   
    ax = plt.subplot(1,3,panel)
    ax.set_title(layer[:4],fontsize=16,weight='bold')
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_facecolor('lightsteelblue')
    ax.spines[['top','right','bottom','left']].set_visible(False)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    africa.plot(ax=ax, color='whitesmoke', edgecolor='white')
    southern_africa.plot(ax=ax, color='lightgrey', edgecolor='white')
    southern_africa.apply(lambda x: ax.annotate(text=x['iso_a3'], xy=x.geometry.centroid.coords[0], ha='center',color='grey'), axis=1)
    for i in range(Nc):
        marker_size = marker_size_max*(mine_sites_df[reference_minerals_columns[i]]/tonnage_max)**0.5
        mine_sites_df.geometry.plot(ax=ax, color=reference_mineral_colors[i], edgecolor='none',markersize=marker_size,alpha=0.7)
        
    if key == 'tonnage':
        tonnage_key = 10**np.arange(1,np.ceil(np.log10(tonnage_max)),1)[::-1]
        Nk = tonnage_key.size
    else:
        Nk = reference_minerals.size
    xk = xl[0] + 0.7*dxl
    yk = yl[0] + np.linspace(0.04*dyl,0.22*dyl,Nk)
    dyk = 0.02*dyl
    yt = yk[-1]+np.diff(yk[-3:-1])
    if key == 'tonnage':
        size_key = marker_size_max*(tonnage_key/tonnage_max)**0.5
        key_df = gpd.GeoDataFrame(geometry=gpd.points_from_xy(np.ones(Nk)*xk, yk))
        key_df.geometry.plot(ax=ax,markersize=size_key,color='k')
        ax.text(xk-0.15*dxl,yt,'Mine annual output (tonnes)',weight='bold',va='center')
        for k in range(Nk):
            ax.text(xk,yk[k],'     {:,.0f}'.format(tonnage_key[k]),va='center')
    if key == 'mineral':
        ax.text(xk-0.03*dxl,yt,'Mineral produced',weight='bold',va='center')
        for k in range(Nk):
            ax.text(xk,yk[k],'   '+reference_minerals[k].capitalize(),va='center')
            ax.plot(xk,yk[k],'s',mfc=reference_mineral_colors[k],mec='lightgrey',ms=10)

    plt.text(xl[0]+0.13*dxl,yk[-4],'Total = {:.1f}\nmillion tonnes'.format(mine_sites_df["total_tons"].sum()/1e6),fontsize=18,weight='bold')
    panel+=1
outfile = 'mine_locations_timeline.png'
plt.savefig(outfile,dpi=300)
print('Created '+outfile)
plt.close(fig)
plt.close()