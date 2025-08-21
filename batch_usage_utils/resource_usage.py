import numpy as np
import pandas as pd
from batch_usage_utils import PipelineMetadata


__all__ = ["ResourceUsage", "fit_resource_func"]


def make_query(dataId):
    items = []
    for k, v in dataId.items():
        if isinstance(v, str):
            items.append(f"{k} == '{v}'")
        else:
            items.append(f"{k} == {v}")
    return " and ".join(items)


class ResourceUsage:
    def __init__(self, md_file, qg_info=None):
        self.md = PipelineMetadata.read_md_files(md_file)
        self.qg_info = qg_info
        self._num_warps = None
        self._task_funcs = {}
        if qg_info is not None:
            self._extract_qg_info()
            self._add_task_funcs()
        self._cpu_time_cache = {}
        self._memory_cache = {}

    def _extract_qg_info(self):
        self._num_warps = {tuple(_[:3]): _[3] for _ in
                           zip(self.qg_info['band'],
                               self.qg_info['tract'],
                               self.qg_info['patch'],
                               self.qg_info['num_warps'])}

    def _add_task_funcs(self):
        task = "assembleDeepCoadd"
        df0 = self.md[task]
        cpu_time = fit_resource_func(df0, "run_cpu_time", xcol="num_warps")
        memory = fit_resource_func(df0, "max_rss", xcol="num_warps")

        def task_func(num_warps):
            return cpu_time(num_warps), memory(num_warps)

        self._task_funcs = {task: task_func}

    def _cpu_time(self, task):
        if task not in self._cpu_time_cache:
            try:
                df0 = self.md[task]
            except KeyError:
                mean_cpu_time = 60.
            else:
                if df0.empty:
                    # Use a default value of 60 sec.
                    mean_cpu_time = 60.
                else:
                    mean_cpu_time = np.mean(df0['run_cpu_time'])
            self._cpu_time_cache[task] = mean_cpu_time
        return float(self._cpu_time_cache[task])

    def _memory(self, task):
        if task not in self._memory_cache:
            try:
                df0 = self.md[task]
            except KeyError:
                mean_max_rss = 4.
            else:
                if df0.empty:
                    mean_max_rss = 4.  # 4GB default
                else:
                    mean_max_rss = np.mean(df0['max_rss'])
            self._memory_cache[task] = mean_max_rss
        return float(self._memory_cache[task])

    def _task_instance_request(self, task, dataId):
        if task in self._task_funcs:
            # Handle tasks like assembleDeepCoadd whose resource
            # requirements depend on info from its prerequisites,
            # e.g., # input warps.
            #print(task, dataId)
            key = dataId['band'], dataId['tract'], dataId['patch']
            num_warps = self._num_warps[key]
            return self._task_funcs[task](num_warps)
        cpu_time = self._cpu_time(task)
        memory = self._memory(task)
        return cpu_time, memory

    def __call__(self, job, prereq_info=None):
        cpu_time_total = 0
        memory_max = 0
        for task, count in job.quanta_counts.items():
            if task in self._task_funcs:
                dataId = job.tags
            else:
                dataId = None
            cpu_time, memory = self._task_instance_request(task, dataId)
            cpu_time_total += count*cpu_time
            memory_max = max(memory, memory_max)
        return cpu_time_total, memory_max


def fit_resource_func(df0, ycol, xcol='num_warps', nbins=50,
                      percentile=95., deg=1):
    df = pd.DataFrame(df0[[xcol, ycol]])
    bins = np.linspace(0, max(df[xcol]), nbins)
    df['bin'] = np.digitize(df[xcol], bins=bins)
    bin_values = sorted(set(df['bin']))
    bin_centers = (bins[1:] + bins[:-1])/2.
    xx = []
    yy = []
    for bin_value, bin_center in zip(bin_values, bin_centers):
        my_df = df.query(f"bin == {bin_value}")
        if my_df.empty:
            continue
        xx.append(float(bin_center))
        yy.append(float(np.percentile(my_df[ycol], percentile)))
    result = np.polyfit(xx, yy, deg)
    return np.poly1d(result)
