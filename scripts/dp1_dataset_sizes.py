import os
import glob
import pickle
from tqdm import tqdm
import pandas as pd
import lsst.daf.butler as daf_butler
from repo_tools import get_dstype_map, get_dataset_sizes

repo = "/repo/main"
parent_col = 'LSSTComCam/runs/DRP/DP1/w_2025_07/DM-48940'
butler = daf_butler.Butler(repo, collections=[parent_col])

dsmap_file = "dp1_dsmap.pickle"
if not os.path.isfile(dsmap_file):
    dsmap = {}
    submit_dir = "/sdf/data/rubin/user/huanlin/comcam/DM-48940/submit/LSSTComCam/runs/DRP/DP1/w_2025_07/DM-48940"
    for i in "1 2a 2b 2c 2d 2e 3a 3b 4 5 6 7".split():
        step = f"step{i}"
        pattern = os.path.join(submit_dir, '*', f'*{step}*.yaml')
        prod_dir = os.path.dirname(sorted(glob.glob(pattern))[-1])
        qg_file = glob.glob(os.path.join(prod_dir, '*.qgraph'))[0]
        print("processing:", qg_file)
        my_dsmap = get_dstype_map(qg_file)
        dsmap.update(my_dsmap)
    with open(dsmap_file, "wb") as fobj:
        pickle.dump(dsmap, fobj)
else:
    with open(dsmap_file, 'rb') as fobj:
        dsmap = pickle.load(fobj)

ds_sizes_file = "dp1_ds_sizes.parquet"
if not os.path.isfile(ds_sizes_file):
    df0 = get_dataset_sizes(butler, dsmap, limit=20000, nsamp=1000)
    df0.to_parquet(ds_sizes_file)
else:
    df0 = pd.read_parquet(ds_sizes_file)

num_refs = []
mean_sizes = []
for _, row in tqdm(df0.iterrows()):
    refs = butler.query_datasets(row.dstype, limit=None)
    num_refs.append(len(refs))
    mean_sizes.append(row['mean']/1024**3)

df0['num_refs'] = num_refs
df0['mean size (GB)'] = mean_sizes
df0['total size (TB)'] = df0['num_refs']*df0['mean size (GB)']/1024.
df0.to_parquet(ds_sizes_file)
