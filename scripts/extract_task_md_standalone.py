#!/usr/bin/env python
"""
Standalone version of task metadata extraction script, i.e., no dependence
on batch_usage_utils.
"""
import os
from collections import defaultdict
from itertools import pairwise
import multiprocessing
import pickle
from graphlib import TopologicalSorter
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy import units as u
from tqdm import tqdm
import lsst.daf.butler as daf_butler
from lsst.pipe.base import Pipeline


class PipelineInfo:
    def __init__(self, pipeline_yaml, repo):
        pipeline = Pipeline.from_uri(pipeline_yaml)
        butler = daf_butler.Butler(repo)
        self.pipeline_graph = pipeline.to_graph(registry=butler.registry)
        self.task_graph = self.pipeline_graph.make_task_xgraph()
        self._task_sequence = None

    @property
    def task_sequence(self):
        if self._task_sequence is None:
            ts = TopologicalSorter()
            for node in self.task_graph:
                ts.add(node, *list(self.task_graph.predecessors(node)))
            ts.prepare()
            self._task_sequence = []
            while ts.is_active():
                jobs = ts.get_ready()
                self._task_sequence.extend(jobs)
                for job in jobs:
                    ts.done(job)
        return self._task_sequence

    def get_dimensions(self, task,
                       ignorable_dims=("group",),
                       replacement_dims=(('exposure', 'visit'),
                                         ('physical_filter', 'band'),
                                         ('healpix3', 'tract'))):
        required_dims = set(
            self.pipeline_graph.tasks[task.name].dimensions.required)
        dimensions = required_dims.difference(ignorable_dims)
        for old_dim, new_dim in replacement_dims:
            if old_dim in dimensions:
                dimensions.remove(old_dim)
                dimensions.add(new_dim)
        return dimensions

    def predecessors(self, task):
        return self.task_graph.predecessors(task)


class PipelineMetadata(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def read_md_files(md_files):
        if isinstance(md_files, str):
            md_files = [md_files]
        dfs = defaultdict(list)
        for md_file in md_files:
            with open(md_file, "rb") as fobj:
                my_dfs = pickle.load(fobj)
                for key in my_dfs:
                    dfs[key].append(my_dfs[key])
        for key in dfs:
            dfs[key] = pd.concat(dfs[key])
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


def extract_metadata(md, data):
    prefixes = ("prep", "init", "start", "end")
    info = md['quantum']
    # Wall times
    times = np.array([Time(info[f"{_}Utc"].split('+')[0])
                      for _ in ("prep", "end")])
    dt = times[1:] - times[:-1]
    for prefix, dt in zip(("run",), dt):
        data[f"{prefix}_wall_time"].append(dt.sec)
    data['start_utc'].append(times[0].mjd)
    data['end_utc'].append(times[-1].mjd)
    # max RSS
    data['max_rss'].append(info['endMaxResidentSetSize']/1024**3)
    # major page faults
    data['major_page_faults'].append(info['endMajorPageFaults'])
    # CPU times
    times = []
    for prefix in prefixes:
        try:
            value = info[f"{prefix}CpuTime"]
        except KeyError:
            value = info["endCpuTime"]
        times.append(value)
    times = np.array(times)
    dt = times[1:] - times[:-1]
    for prefix, dt in zip(("prep", "init", "run"), dt):
        data[f"{prefix}_cpu_time"].append(dt)
    # Node info
    if 'nodeName' in info:
        data['node_name'].append(info['nodeName'])
    # Butler put/get info
    if 'butler_metrics' in info:
        for key in ('time_in_put', 'time_in_get', 'n_get', 'n_put'):
            data[key].append(info['butler_metrics'][key])
    else:
        for key in ('time_in_put', 'time_in_get', 'n_get', 'n_put'):
            data[key].append(None)
    return data


def extract_from_provenance_graph(butler, refs, disable_progress=True):
    data = defaultdict(list)
    id_map = defaultdict(set)
    ref_map = {}
    for ref in refs:
        id_map[ref.run].add(ref.id)
        ref_map[ref.id] = ref
    for collection, ids in tqdm(id_map.items(), disable=disable_progress):
        md_dict = butler.get("run_provenance.metadata",
                             parameters={"datasets": ids},
                             collections=[collection])
        for _id, md_attempts in md_dict.items():
            ref = ref_map[_id]
            try:
                extract_metadata(md_attempts[-1], data)
            except Exception as eobj:
                raise eobj
            else:
                dataId = (ref.dataId.to_simple()  # noqa:N806
                          .model_dump()['dataId'])
                for key, value in dataId.items():
                    data[key].append(value)
    return pd.DataFrame(data), []


def extract_from_refs(butler, refs, disable_progress=True):
    data = defaultdict(list)
    missing_files = []
    for ref in tqdm(refs, disable=disable_progress):
        md = butler.get(ref)
        try:
            extract_metadata(md, data)
        except Exception as eobj:
            raise eobj
        else:
            dataId = ref.dataId.to_simple().model_dump()['dataId']  # noqa:N806
            for key, value in dataId.items():
                data[key].append(value)
    return pd.DataFrame(data), missing_files


def extract_md_files(repo, collection, tasks, outfile=None, where="",
                     nproc=10, extraction_func=extract_from_provenance_graph,
                     partition_factor=1):
    butler = daf_butler.Butler(repo, collections=[collection])
    if outfile is None:
        test_name = collection.split("/")[2]
        outfile = f"{test_name}.pickle"
    if os.path.isfile(outfile):
        dfs = PipelineMetadata.read_md_files([outfile])
    else:
        dfs = PipelineMetadata()
    for task in tasks:
        print(task, flush=True)
        dstype = f"{task}_metadata"
        try:
            refs = list(set(butler.registry.queryDatasets(dstype, where=where,
                                                          findFirst=True)))
            print(len(refs), "refs")
        except (daf_butler.MissingDatasetTypeError,
                daf_butler.EmptyQueryResultError):
            continue
        if task in dfs and len(dfs[task]) == len(refs):
            print(f"   skipping {task}")
            continue
        if nproc == 1:
            df, missing_files = extraction_func(butler, refs)
            df_list = [df]
        else:
            missing_files = []
            df_list = []
            ntranches = nproc * partition_factor
            indices = np.linspace(0, len(refs), ntranches, dtype=int)
            with multiprocessing.Pool(processes=nproc) as pool:
                workers = []
                for imin, imax in pairwise(indices):
                    disable_progress = (imin != 0)
                    workers.append(
                        pool.apply_async(
                            extraction_func, (butler, refs[imin:imax]),
                            {"disable_progress": disable_progress}
                        )
                    )
                pool.close()
                pool.join()
                for worker in workers:
                    df, missing = worker.get()
                    df_list.append(df)
                    missing_files.extend(missing)
        print(f"{len(missing_files)} missing files")
        dfs[task] = pd.concat(df_list)
        # Do an incremental write to avoid reprocessing finished tasks
        # on restart.
        if not dfs:
            return
        dfs.write_md_file(outfile, clobber=True)
    return dfs


def get_task_subsets(pipeline_yaml=None, repo="dp2_prep"):
    if pipeline_yaml is None:
        pipeline_yaml = os.path.join(os.environ["DRP_PIPE_DIR"],
                                     "pipelines", "LSSTCam",
                                     "DRP.yaml")
    pipeline_info = PipelineInfo(pipeline_yaml, repo)
    pg = pipeline_info.pipeline_graph
    stages = sorted(_ for _ in pg.task_subsets.keys() if _.startswith("stage"))

    tasks = {stage[:len("stage1")]: list(pg.task_subsets[stage])
             for stage in stages}
    return tasks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="Data repository")
    parser.add_argument("collection",
                        help=("Collection containing the metadata"))
    parser.add_argument("--stage", default=None,
                        help=("Pipeline stage or step to extract. If None, "
                              "then all stages/steps will be extracted, with "
                              "one pickle file for each."))
    parser.add_argument("--nproc", default=1,
                        help="Number of processes to use for multiprocessing")

    args = parser.parser_args()

    repo = args.repo
    collection = args.collection
    tasks = get_task_subsets(repo=repo)
    if args.stage is None:
        # extract for all stages/steps in the pipeline"
        stages = [_ for _ in tasks if _.startswith("st")]
    else:
        stages = [args.stage]

    butler = daf_butler.Butler(repo)

    for stage in stages:
        print("Working on {stage}")
        extract_md_files(repo, collection, tasks[stage],
                         outfile=f"{stage}_task_md.pickle",
                         nproc=args.nproc)
