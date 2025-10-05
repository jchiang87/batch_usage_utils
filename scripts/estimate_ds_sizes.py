import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import lsst.daf.butler as daf_butler
from lsst.pipe.base import Pipeline

limit = None
nsamp = 1000
not_found_max = 1000

pipeline_yaml = os.path.join(os.environ['DRP_PIPE_DIR'], "pipelines",
                             "LSSTCam", "DRP.yaml")
repo = "/repo/main"
collections = ["LSSTCam/runs/DRP/20250604_20250906/w_2025_37/DM-52496"]

pipeline = Pipeline.from_uri(pipeline_yaml)
butler = daf_butler.Butler(repo, collections=collections)

pg = pipeline.to_graph(registry=butler.registry)

data = defaultdict(list)
for task in pg.tasks:
    for output in pg.tasks[task].outputs.values():
        dstype = output.dataset_type_name
        print(task, dstype, flush=True)
        try:
            refs = butler.query_datasets(dstype, limit=limit)
            print(len(refs), flush=True)
        except daf_butler.EmptyQueryResultError:
            continue
        if nsamp > len(refs):
            sample_refs = refs
        else:
            sample_refs = np.random.choice(refs, replace=False, size=nsamp)
        sizes = []
        files_not_found = 0
        for ref in tqdm(sample_refs):
            try:
                sizes.append(butler.getURI(ref).size())
            except FileNotFoundError:
                files_not_found += 1
            if files_not_found > not_found_max:
                break
        if not sizes:
            continue
        dimensions = pg.tasks[task].dimensions.required
        data['task'].append(task)
        data['dstype'].append(dstype)
        data['numrefs'].append(len(refs))
        data['mean'].append(np.mean(sizes))
        data['median'].append(np.median(sizes))
        data['max'].append(np.max(sizes))
        data['dimensions'].append(",".join(dimensions))
df0 = pd.DataFrame(data)
df0.to_parquet("DM-52496_ds_sizes.parquet")
