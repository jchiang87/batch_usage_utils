from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from .visit_counts import VisitCounts
from .estimated_workflow import PipelineInfo

__all__ = ["cpu_time_summary"]


def num_cores(memory, mem_per_core=6.0):
    if memory < mem_per_core:
        return 1.0
    return np.ceil(memory/mem_per_core)


def relevant_dimensions(dimensions):
    my_dims = sorted(dimensions)
    if "instrument" in my_dims:
        my_dims.remove("instrument")
    if "skymap" in my_dims:
        my_dims.remove("skymap")
    if 'exposure' in my_dims:
        my_dims.remove('exposure')
        my_dims.append('visit')
    if 'physical_filter' in my_dims:
        my_dims.remove('physical_filter')
        my_dims.append('band')
    return my_dims


def cpu_time_summary(overlap_file, stage_yamls, resource_usage,
                     repo="dp2_prep", outfile=None, verbose=False):
    cpu_time = defaultdict(lambda: 0)
    job_count = defaultdict(lambda: 0)
    overlaps = pd.read_parquet(overlap_file)
    vc = VisitCounts(overlap_file)
    for stage, stage_yaml in stage_yamls.items():
        if verbose:
            print(stage, flush=True)
        pipeline_info = PipelineInfo(stage_yaml, repo)
        for task_node_key in tqdm(pipeline_info.task_sequence,
                                  disable=not verbose):
            task = task_node_key.name
            dimensions = relevant_dimensions(
                pipeline_info.get_dimensions(task_node_key))
            df = overlaps[dimensions].drop_duplicates()
            if task in resource_usage._task_funcs:
                # Compute per instance resource usage.
                if 'band' not in df:
                    df = pd.DataFrame(df)
                    df['band'] = None
                for args in zip(df['patch'], df['tract'], df['band']):
                    num_visits = vc(*args)
                    job_cpu_time, job_memory = resource_usage(task, num_visits)
                    cpu_time[task] += num_cores(job_memory)*job_cpu_time
                    job_count[task] += 1
            else:
                # Can use the mean values for instances.
                if not dimensions:
                    num_jobs = 1
                else:
                    num_jobs = len(df)
                job_cpu_time, job_memory = resource_usage(task, None)
                cpu_time[task] += num_jobs*num_cores(job_memory)*job_cpu_time
                job_count[task] += num_jobs
    tasks = list(cpu_time.keys())
    df0 = pd.DataFrame(dict(task=tasks,
                            cpu_time=list(cpu_time.values()),
                            num_jobs=[job_count[_] for _ in tasks]))
    if outfile is not None:
        df0.to_parquet(outfile)
    return df0
