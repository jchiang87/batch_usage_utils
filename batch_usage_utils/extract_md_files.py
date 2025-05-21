import os
from collections import defaultdict
import pickle
import subprocess
import numpy as np
import pandas as pd
from astropy import units as u
from tqdm import tqdm
import lsst.daf.butler as daf_butler
from batch_usage_utils import extract_md_json


__all__ = ["extract_md_files", "PipelineMetadata"]


class PipelineMetadata(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def read_md_files(md_files):
        if isinstance(md_files, str):
            md_files = [md_files]
        dfs = {}
        for md_file in md_files:
            with open(md_file, "rb") as fobj:
                dfs.update(pickle.load(fobj))
        return PipelineMetadata(dfs)

    def write_md_file(self, md_file, clobber=False):
        if os.path.isfile(md_file) and not clobber:
            raise FileExistsError("{md_file} exists and clobber=False")
        with open(md_file, "wb") as fobj:
            pickle.dump(dict(self), fobj)

    def projected_wall_times(self, ds_sizes=None, num_cores=10000,
                             runtime_column='run_wall_time'):
        total_projected = 0
        total_storage = 0
        data = defaultdict(list)
        for task, df in self.items():
            data['task'].append(task)
            data['max_memory'].append(np.max(df['max_rss']))
            max_wt = np.max(df[runtime_column])/3600.*u.h
            data['max_wall_time'].append(max_wt)
            data['median_wall_time'].append(
                np.median(df[runtime_column])/3600.*u.h)
            mean_wt = np.mean(df[runtime_column]/3600)*u.h
            data['mean_wall_time'].append(mean_wt)
            num_jobs = len(df)
            data['num_jobs'].append(num_jobs)
            total = num_jobs*mean_wt
            projected = max(max_wt, total/min(num_cores, num_jobs))
            data['projected'].append(projected.copy())
            total_projected += projected
            print(total_projected)
            data['total_projected'].append(total_projected.copy())
            if ds_sizes is not None:
                storage = num_jobs*ds_sizes[task]/1024**4*u.terabyte
                data['storage'].append(storage.copy())
                total_storage += storage
                data['total_storage'].append(total_storage.copy())
        return pd.DataFrame(data)


def extract_md_files(repo, collection, tasks, outfile=None,
                     local_dir="./replicas", where="", downloaded=()):
    butler = daf_butler.Butler(repo, collections=[collection])
    if outfile is None:
        test_name = collection.split("/")[2]
        outfile = f"{test_name}.pickle"
    os.makedirs(local_dir, exist_ok=True)
    if os.path.isfile(outfile):
        dfs = PipelineMetadata.read_md_files([outfile])
    else:
        dfs = PipelineMetadata()
    for task in tasks:
        if task in dfs:
            continue
        print(task, flush=True)
        dstype = f"{task}_metadata"
        if task not in downloaded:
            command = (f"butler retrieve-artifacts --dataset-type {dstype} "
                       "--find-first "
                       f"--collections {collection} --clobber {repo} "
                       f"{local_dir} --limit 0")
            if where:
                command += f'--where "{where}"'
            print(command)
            try:
                subprocess.check_call(command, shell=True)
            except subprocess.CalledProcessError:
                pass
        data = defaultdict(list)
        try:
            refs = list(set(butler.registry.queryDatasets(dstype, where=where,
                                                          findFirst=True)))
            print(len(refs), "refs")
        except (daf_butler.MissingDatasetTypeError,
                daf_butler.EmptyQueryResultError):
            continue
        missing_files = []
        for ref in tqdm(refs):
            try:
                md_file = local_dir + butler.getURI(ref).path
                data = extract_md_json(md_file, data)
            except FileNotFoundError:
                missing_files.append(ref)
            else:
                dataId = ref.dataId.to_simple().model_dump()['dataId']
                for key, value in dataId.items():
                    data[key].append(value)
        print(f"{len(missing_files)} missing files")
        dfs[task] = pd.DataFrame(data)
        # Do an incremental write to avoid reprocessing finished tasks
        # on restart.
        if not dfs:
            return
        dfs.write_md_file(outfile, clobber=True)
    return dfs
