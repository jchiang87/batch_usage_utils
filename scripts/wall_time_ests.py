import os
import glob
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

DS_SIZES = defaultdict(lambda : 0)
df0 = pd.read_parquet("dataset_sizes.parquet")
for _, row in df0.iterrows():
    DS_SIZES[row.task] += row['mean']

def projected_wall_times(md_file, total_projected=0, factor=1,
                         runtime_column='run_wall_time', num_cores=8640.,
                         ds_sizes=DS_SIZES, total_storage=0):
    with open(md_file, "rb") as fobj:
        dfs = pickle.load(fobj)
    for task, df in dfs.items():
        max_wt = np.max(df[runtime_column])/3600.
        median_wt = np.median(df[runtime_column])/3600.
        mean_wt = np.mean(df[runtime_column])/3600.
        max_mem = np.max(df['max_rss'])
        num_jobs = factor*len(df)
        total = num_jobs*mean_wt
        projected = max(max_wt, total/min(num_cores, num_jobs))
        storage = num_jobs*DS_SIZES[task]/1024**4
        total_storage += storage
        total_projected += projected
        print(f"{task:39s}{max_mem:5.1f}{max_wt:8.3f}{median_wt:8.3f}{mean_wt:8.3f}{num_jobs:10d}{projected:8.2f}{total_projected:8.2f}{storage:8.1f}{total_storage:8.1f}")
    print()
    return total_projected, total_storage

md_files = sorted(glob.glob("step*_md.pickle"))

print("task                                  max_mem       wall time (h)       # jobs   projected (h)    storage (TB) ")
print("                                        (GB)    max   median    mean              task   total   task    total")
total_projected = 0
total_storage = 0
for md_file in md_files:
    step = md_file.split('_')[0]
    print(step)
    if step[-1] in '345':
        factor = 10
    else:
        factor = 1
    total_projected, total_storage = projected_wall_times(md_file, total_projected=total_projected, total_storage=total_storage, factor=factor)
