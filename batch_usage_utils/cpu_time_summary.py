from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from .visit_counts import VisitCounts
from .estimated_workflow import PipelineInfo


__all__ = ["cpu_time_summary"]


USDF_PARTITIONS = {
    "milano": dict(num_nodes=110,
                   cores_per_node=120,
                   mem_per_core=4.0,
                   speedup=1.0),
    "torino": dict(num_nodes=35,
                   cores_per_node=120,
                   mem_per_core=6.0,
                   speedup=2.0)
}


class NodeServer:
    def __init__(self, partitions=USDF_PARTITIONS):
        self.partitions = partitions
        self.node_types = []
        weights = []
        for partition, config in partitions.items():
            self.node_types.append(partition)
            weights.append(np.prod(list(config.values())))
        weights = np.array(weights)
        self.weights = weights/sum(weights)

    def resources(self, cpu_time, memory, node_type=None):
        if node_type is None:
            node_type = np.random.choice(self.node_types, p=self.weights)
        config = self.partitions[node_type]
        cpu = cpu_time/config['speedup']
        num_cores = int(np.ceil(memory/config['mem_per_core']))
        return cpu, num_cores, node_type


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


def cpu_time_summary(overlaps, visit_counts, stage_yamls, resource_usage,
                     partitions=USDF_PARTITIONS,
                     repo="dp2_prep", outfile=None, verbose=False):
    for config in partitions.values():
        config['num_cores'] = config['num_nodes']*config['cores_per_node']
    node_server = NodeServer(partitions=partitions)
    node_type_cpu_time = defaultdict(lambda : defaultdict(lambda: 0))
    job_count = defaultdict(lambda: 0)
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
                    num_visits = visit_counts(*args)
                    job_cpu_time, num_cores, node_type = node_server.resources(
                        *resource_usage(task, num_visits))
                    node_type_cpu_time[node_type][task] += num_cores*job_cpu_time
                    job_count[task] += 1
            else:
                # Can use the mean values for instances.
                if not dimensions:
                    num_jobs = 1
                else:
                    num_jobs = len(df)
                for node_type, weight in zip(node_server.node_types,
                                             node_server.weights):
                    job_cpu_time, num_cores, node_type = node_server.resources(
                        *resource_usage(task, None), node_type=node_type)
                    node_type_cpu_time[node_type][task] += weight*num_jobs*num_cores*job_cpu_time
                job_count[task] += num_jobs
    wall_time = defaultdict(lambda : 0)
    cpu_time = defaultdict(lambda : 0)
    for node_type in node_type_cpu_time:
        for task, value in node_type_cpu_time[node_type].items():
            cpu_time[task] += value
            wall_time_est = value/partitions[node_type]["num_cores"]
            if wall_time_est > wall_time[task]:
                wall_time[task] = wall_time_est
    tasks = list(cpu_time.keys())
    df0 = pd.DataFrame(dict(task=tasks,
                            cpu_time=list(cpu_time.values()),
                            wall_time=list(wall_time.values()),
                            num_jobs=[job_count[_] for _ in tasks]))
    if outfile is not None:
        df0.to_parquet(outfile)
    return df0
