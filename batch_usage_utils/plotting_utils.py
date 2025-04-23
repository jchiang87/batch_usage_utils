import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["scatter_plot_by_node"]

def scatter_plot_by_node(df0, title=None, xcol='run_cpu_time', ycol='max_rss',
                         ra_node_prefix='s-lsstcam-run-sfm-runner-workerset-'):
    # plot ycol vs xcol disaggregated by node name
    nodes = sorted(set(df0['node_name']))
    # plot all RA processings as one group
    df = df0[df0['node_name'].str.startswith(ra_node_prefix)]
    plt.scatter(df[xcol], df[ycol], s=2, label=f'{ra_node_prefix}-*')
    for node_name in nodes:
        if not node_name.startswith('sdf'):
            continue
        df = df0.query(f"node_name=='{node_name}'")
        plt.scatter(df[xcol], df[ycol], s=2, label=node_name)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.legend(fontsize='x-small', ncol=2)
    plt.title(title)
