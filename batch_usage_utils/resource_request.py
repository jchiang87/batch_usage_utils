import numpy as np
from batch_usage_utils import PipelineMetadata


__all__ = ["ResourceRequest"]


def make_query(dataId):
    components = []
    for k, v in dataId.items():
        if isinstance(v, str):
            components.append(f"{k}=='{v}'")
        else:
            components.append(f"{k}=={v}")
    return ' and '.join(components)


class ResourceRequest:
    def __init__(self, md_file):
        self.md = PipelineMetadata.read_md_files(md_file)
        self._cpu_time_cache = {}
        self._memory_cache = {}

    def _cpu_time(self, task):
        if task not in self._cpu_time_cache:
            df0 = self.md[task]
            if df0.empty:
                # Use a default value
                mean_cpu_time = 60. # 60 seconds
            else:
                mean_cpu_time = np.mean(df0['run_cpu_time'])
            self._cpu_time_cache[task] = mean_cpu_time
        return float(self._cpu_time_cache[task])

    def _memory(self, task):
        if task not in self._memory_cache:
            df0 = self.md[task]
            if df0.empty:
                mean_max_rss = 4.  # 4GB default
            else:
                mean_max_rss = np.mean(df0['max_rss'])
            self._memory_cache[task] = mean_max_rss
        return float(self._memory_cache[task])

    def _task_instance_request(self, task, prereq_info=None):
        # ToDo:  Add a branch that handles assembleDeepCoadd et al.
        cpu_time = self._cpu_time(task)
        memory = self._memory(task)
        return cpu_time, memory

    def __call__(self, job, prereq_info=None):
        cpu_time_total = 0
        memory_max = 0
        for task, count in job.gwf_job.quanta_counts.items():
            cpu_time, memory = self._task_instance_request(task)
            cpu_time_total += cpu_time
            memory_max = max(memory, memory_max)
        return cpu_time_total, memory_max

if __name__ == '__main__':
    #from get_coadd_dependency_info import *
    from get_parsl_graph import graph

    md_file = ("/sdf/data/rubin/user/jchiang/on-sky/task_metadata/DRP/"
               "DRP_20250420_20250429_md.pickle")
    resource_request = ResourceRequest(md_file)

    #job_names = graph.get_jobs("makeWarpTract")
    #job_names = graph.get_jobs("coadd")
    job_names = graph.get_jobs("plotPropertyMapSurvey")

    for job_name in job_names[:10]:
        job = graph[job_name]
        print(job.gwf_job.label, job.gwf_job.tags,
              job.gwf_job.quanta_counts,
              resource_request(job))
