import os
from graphlib import TopologicalSorter
from tqdm import tqdm
import numpy as np
import pandas as pd
import lsst.daf.butler as daf_butler
from lsst.pipe.base import Pipeline
from .workflow import Job, Workflow


__all__ = ["EstimatedWorkflow"]


class PipelineInfo:
    def __init__(self, pipeline_yaml, repo="/repo/main"):
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


class EstimatedWorkflow(Workflow):
    def __init__(self):
        super().__init__()

    def __getitem__(self, job_id):
        if job_id not in self:
            self.add_job(job_id)
        return dict.__getitem__(self, job_id)

    def add_job(self, job_id):
        job = Job()
        job.id = job_id
        job.task_label = job_id[0]
        job.dataId = dict(job_id[1])
        job.quanta_counts = {job.task_label: 1}
        self[job_id] = job

    @staticmethod
    def build_from_overlaps(overlaps, pipeline_yaml, repo="/repo/main"):
        pipeline_info = PipelineInfo(pipeline_yaml, repo=repo)

        # Add skymap name and instrument columns in case they aren't present.
        overlaps['skymap'] = 'lsst_cells_v1'
        overlaps['instrument'] = 'LSSTCam'

        wf = EstimatedWorkflow()
        for task in pipeline_info.task_sequence:
            print(task)
            dimensions = sorted(pipeline_info.get_dimensions(task))
            upstream_tasks = list(pipeline_info.predecessors(task))
            for upstream_task in upstream_tasks:
                upstream_dims = pipeline_info.get_dimensions(upstream_task)
                relevant_dims = sorted(upstream_dims.union(dimensions))
                upstream_dims = sorted(upstream_dims)
                df = overlaps[relevant_dims].drop_duplicates()
                for _, row in tqdm(df.iterrows(), total=len(df)):
                    job_id = (task.name,
                              tuple([(dim, row[dim]) for dim in dimensions]))
                    upstream_id = (upstream_task.name,
                                   tuple([(dim, row[dim]) for dim
                                          in upstream_dims]))
                    wf[job_id].predecessors.add(wf[upstream_id].id)
        return wf


if __name__ == '__main__':
    overlaps_file = "DM-51933_visit_skymap_overlaps.parquet"
    pipeline_yaml = os.path.join(os.environ['DRP_PIPE_DIR'],
                                 "pipelines", "LSSTCam", "DRP.yaml")

    wf = EstimatedWorkflow.build_from_overlaps(overlaps_file, pipeline_yaml)
    wf.save("DM-51933_approx_wf.pickle")
