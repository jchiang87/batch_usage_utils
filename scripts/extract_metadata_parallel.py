import os
import sys
from collections import defaultdict
from itertools import pairwise
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import lsst.daf.butler as daf_butler
from batch_usage_utils import PipelineMetadata, extract_md_json,\
    extract_metadata, PipelineInfo


def extract_from_refs(butler, refs):
    data = defaultdict(list)
    missing_files = []
    for ref in refs:
        md = butler.get(ref)
        try:
            extract_metadata(md, data)
        except Exception as eobj:
            raise eobj
#        try:
#            md_file = butler.getURI(ref).path
#            data = extract_md_json(md_file, data)
#        except FileNotFoundError:
#            missing_files.append(ref)
        else:
            dataId = ref.dataId.to_simple().model_dump()['dataId']  # noqa:N806
            for key, value in dataId.items():
                data[key].append(value)
    return pd.DataFrame(data), missing_files


def extract_md_files(repo, collection, tasks, outfile=None, where="",
                     nproc=10):
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
            df, missing_files = extract_from_refs(butler, refs)
            df_list = [df]
        else:
            missing_files = []
            df_list = []
            indices = np.linspace(0, len(refs), nproc, dtype=int)
            with multiprocessing.Pool(processes=nproc) as pool:
                workers = []
                for imin, imax in pairwise(indices):
                    workers.append(pool.apply_async(
                        extract_from_refs, (butler, refs[imin:imax])))
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
    pipeline_info = PipelineInfo(pipeline_yaml)
    pg = pipeline_info.pipeline_graph
    stages = sorted(_ for _ in pg.task_subsets.keys() if _.startswith("stage"))

    tasks = {stage[:len("stage1")]: list(pg.task_subsets[stage]) for stage in stages}
    return tasks


if __name__ == "__main__":
    repo = "/repo/main"
    tasks = get_task_subsets(repo=repo)
    butler = daf_butler.Butler(repo)

    collection = "LSSTCam/runs/DRP/20250421_20250921/w_2025_41/DM-52836"

    stage = sys.argv[1]
    assert stage in tasks

    extract_md_files(repo, collection, tasks[stage],
                     outfile=f"{stage}_task_md.pickle",
                     nproc=32)
