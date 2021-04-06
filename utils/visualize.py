"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to visualize all steps of the analysis
"""

import matplotlib
import numpy as np
import pandas as pd
from nilearn import plotting

def plot_connectivity(X_conn):
    regions = ['LF','LC','LP','LO','LT','RF','RC','RP','RO','RT']
    try:
        coords = np.loadtxt('../utils/coordinates.txt')
    except:
        coords = np.loadtxt('utils/coordinates.txt')

    for t in range(len(X_conn)):
        tmp = X_conn
        conn_tmp = pd.DataFrame(np.zeros((len(regions), len(regions))))
        conn_tmp.columns = regions
        conn_tmp.index = regions

        for i in regions:
            for a in regions:
                try:
                    conn_tmp.loc[i, a] = tmp[i + '_' + a][0]
                except:
                    conn_tmp.loc[i, a] = tmp[a + '_' + i][0]

    conn_matrix = np.array(conn_tmp)

    colormap = matplotlib.cm.get_cmap('YlOrRd')
    fig = plotting.plot_connectome(conn_matrix, node_coords=coords,
                                   edge_cmap=colormap, colorbar=True, edge_vmin=0, edge_vmax=0.8,
                                   node_color=colormap(conn_matrix.diagonal()),
                                   display_mode='lzr')


    """    colormap = matplotlib.cm.get_cmap('OrRd')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        fig = plotting.plot_connectome(conn_matrix, node_coords=coords, edge_vmin=0, edge_vmax=1,
                                       edge_cmap=colormap, colorbar=True, edge_threshold=None,
                                       node_color=colormap(norm(conn_matrix.diagonal())),
                                       display_mode='lzr')
    """
    return fig

