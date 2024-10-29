# -*- coding: utf-8 -*-
"""Employee Review Analysis - GSOM Mapping - Discourse Analysis Dimensions

***Install and import packages***
"""

!pip install pygsom

import pandas as pd
import gsom

"""***Data loading***"""

df = pd.read_excel('File-Name.xlsx')
df.head()

"""***Feature selection***"""

data_training = df.iloc[:, 1:16]
data_training.columns

"""***Visualizing variable distributions***"""

import warnings

# ignore warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

# Visualize the distribution of each variable.
plt.figure(figsize=(15,30))
for i, j in enumerate(data_training.describe().columns):
    plt.subplot(12,5, i+1)
    sns.distplot(x=data_training[j])
    plt.xlabel(j)
    plt.title('{} Distribution'.format(j))
    # plt.subplots_adjust(wspace=.2, hspace=.5)
    plt.tight_layout()
plt.show()

"""***Training GSOM***

***Change feature count***
"""

gsom_map = gsom.GSOM(0.83, 15, max_radius=4)
gsom_map.fit(data_training.to_numpy(), 100, 50)

"""***Predict using GSOM***"""

df_all=pd.DataFrame(data_training.to_numpy(), columns=data_training.columns)
df_all["uid"]=df["ID"].values
df_all["data-title"]=df["ID"].values

"""***Visualization***"""

map_points = gsom_map.predict(df_all,"uid","data-title")
gsom.plot(map_points, "data-title", gsom_map=gsom_map, figure_label='IB_Analysis', file_name='IB_Analysis',show_index=False,cmap_colors="Paired")
map_points.to_csv("GSOM_IB.csv", index=False)

"""***Mapping GSOM points to IDs***"""

def get_mappoint_id_mapping(map_points):
    output_header = ['uid','x','y', 'output', 'hit_count']
    output_data = []

    for row in map_points.itertuples():
        #output_header.append(row.output)
        id_array = row.uid
        for idx, x in enumerate(id_array):
            output_data.append([x, row.x, row.y, row.output, row.hit_count])

    df_out = pd.DataFrame(output_data, columns=output_header)
    return df_out

"""***Viewing predicted map points***"""

map_points.head()

"""***Saving to CSV file***"""

output_df = get_mappoint_id_mapping(map_points)
output_df.to_csv('mappoint_id_mapping.csv', index=False)

"""***Get GSOM map weights***"""

df_weights = pd.DataFrame(gsom_map.node_list)
df_weights.head()

