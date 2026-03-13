import os
import sys
from collections import defaultdict
from itertools import pairwise
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
import lsst.daf.butler as daf_butler
from batch_usage_utils import PipelineMetadata, extract_md_json, \
    extract_metadata, PipelineInfo


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

    tasks = {stage[:len("stage1")]: list(pg.task_subsets[stage]) for stage in stages}
    return tasks


if __name__ == "__main__":
    tasks = get_task_subsets()
    repo = "dp2_prep"
    butler = daf_butler.Butler(repo)

    stage = sys.argv[1]
    assert stage in tasks

    collection = f"LSSTCam/runs/DRP/DP2/v30_0_0/DM-53881/{stage}"

    extract_md_files(repo, collection, tasks[stage],
                     outfile=f"{stage}_task_md.pickle",
                     nproc=32)
