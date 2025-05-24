"""Module to map uuids to dataIds from QuantumGraphs"""

import os
import glob
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import lsst.daf.butler as daf_butler
import lsst.pipe.base as pipe_base


__all__ = ["QGExtract", "get_run_collections", "quantum_graph_file"]


class QGExtract:
    def __init__(self, qg_file):
        self.qg = pipe_base.QuantumGraph.loadUri(qg_file)
        self._set_default_node_maps()

    def update_task_node_func_maps(self, node_func_maps):
        self._task_node_func_maps.update(node_func_maps)

    def map_uuids(self, tasks=None):
        data = defaultdict(list)
        for node in tqdm(self.qg.graph.nodes):
            task = node.taskDef.label
            if tasks is not None and task not in tasks:
                continue
            uuid = str(node.nodeId)
            dataId = node.quantum.dataId  # noqa: N806
            dimensions = dataId.dimensions.to_simple()
            data["uuid"].append(uuid)
            for dimension in dimensions:
                data[dimension].append(dataId[dimension])
            self._add_task_specific_info(node, data, task)
        return pd.DataFrame(data)

    def _add_task_specific_info(self, node, data, task):
        if task not in self._task_node_func_maps:
            return
        node_func_maps = self._task_node_func_maps[task]
        for column, node_func in node_func_maps.items():
            data[column].append(node_func(node))

    def _set_default_node_maps(self):
        self._task_node_func_maps = {
            "assembleDeepCoadd": {
                "num_warps": lambda node: len(
                    node.quantum.inputs["direct_warp"]
                )
            }
        }


def get_run_collections(repo, parent_collection, target_task):
    butler = daf_butler.Butler(repo)
    run_collections = sorted(
        butler.registry.queryCollections(
            f"{parent_collection}/202*", collectionTypes=daf_butler.CollectionType.RUN
        )
    )
    target_collections = []
    for run_collection in run_collections:
        try:
            butler.registry.queryDatasets(f"{target_task}_metadata", collections=run_collection)
        except daf_butler.EmptyQueryResultError:
            continue
        else:
            target_collections.append(run_collection)
    return target_collections


S3DF_QG_FILE_ROOT = "/sdf/data/rubin/panda_jobs/panda_cache/panda_cache_box/payload"


def quantum_graph_file(run_collection, qg_file_root=S3DF_QG_FILE_ROOT):
    folder = run_collection.replace("/", "_")
    pattern = os.path.join(qg_file_root, folder, "*.qgraph")
    try:
        file_path = glob.glob(pattern)[0]
    except IndexError:
        raise RuntimeError("QG file not found globbing {pattern}")
    return file_path


if __name__ == "__main__":
    repo = "/repo/main"
    parent_collection = "LSSTComCam/runs/DRP/DP1/w_2025_04/DM-48556"

    task = "assembleCoadd"
    run_collections = get_run_collections(repo, parent_collection, task)

    dfs = {}
    for run_collection in run_collections:
        print(run_collection, flush=True)
        qg_file = quantum_graph_file(run_collection)

        qg_extract = QGExtract(qg_file)
        dfs[run_collection] = qg_extract.map_uuids(tasks=[task])
