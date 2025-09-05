from types import MappingProxyType
import numpy as np
from scipy.stats import binned_statistic
from batch_usage_utils import PipelineMetadata


__all__ = ["ResourceUsage"]


def make_query(dataId):  # noqa:N803
    items = []
    for k, v in dataId.items():
        if isinstance(v, str):
            items.append(f"{k} == '{v}'")
        else:
            items.append(f"{k} == {v}")
    return " and ".join(items)


class UseWarpFunc(dict):
    def __init__(self, cpu_time, memory):
        super().__init__()
        self['run_cpu_time'] = cpu_time
        self['max_rss'] = memory


RESOURCE_DEFAULTS = MappingProxyType({"run_cpu_time": 60, "max_rss": 4.0})
RESOURCE_FIT_PERCENTILES = MappingProxyType({"run_cpu_time": 50, "max_rss": 95})

WARP_INDEX_COLS = ["tract", "patch", "band"]

NUM_WARP_TASKS = {
    'selectDeepCoaddVisits': UseWarpFunc(True, True),
    'selectTemplateCoaddVisits': UseWarpFunc(True, True),
    'assembleDeepCoadd': UseWarpFunc(True, True),
    'assembleTemplateCoadd': UseWarpFunc(True, True),
    'assembleCellCoadd': UseWarpFunc(True, True),
    'detectCoaddPeaks': UseWarpFunc(False, True),
    'deconvolve': UseWarpFunc(False, True),
    'deblendCoaddFootprints': UseWarpFunc(True, True),
    'measureObjectUnforced': UseWarpFunc(True, True),
    'fitDeepCoaddPsfGaussians': UseWarpFunc(True, True),
    'measureObjectForced': UseWarpFunc(True, True),
    'associateDiaSource': UseWarpFunc(True, False),
    'calculateDiaObject': UseWarpFunc(True, False),
    'standardizeObjectForcedSource': UseWarpFunc(True, False),
    'splitPrimaryObjectForcedSource': UseWarpFunc(True, False),
    'standardizeDiaObjectForcedSource': UseWarpFunc(True, False),
}


class ResourceUsage:
    def __init__(self, md_files, df_warps=None):
        self.md = PipelineMetadata.read_md_files(md_files)
        self.df_warps = df_warps
        if df_warps is not None:
            self._set_num_warps()
        # Add specialized resource request functions.
        self._task_funcs = {}
        if df_warps is not None:
            self._add_task_funcs()
        self._resource_cache = {}

    def _set_num_warps(self):
        df = self.df_warps
        self._num_warps = dict(zip(zip(df["tract"], df["patch"], df["band"]),
                                   df["num_warps"]))
        df1 = df.groupby(["tract", "patch"])["num_warps"].sum().reset_index()
        self._num_warps1 = dict(zip(zip(df1["tract"], df1["patch"]),
                                    df1["num_warps"]))

    def num_warps(self, tract, patch, band=None):
        if band is None:
            return self._num_warps1[(tract, patch)]
        return self._num_warps[(tract, patch, band)]

    def _add_task_funcs(self):
        for task, func_status in NUM_WARP_TASKS.items():
            task_funcs = {}
            for column in RESOURCE_DEFAULTS:
                if func_status[column]:
                    percentile = RESOURCE_FIT_PERCENTILES[column]
                    task_funcs[column] \
                        = self.fit_resource_func(task, column,
                                                 percentile=percentile)
                else:
                    task_funcs[column] \
                        = lambda x: self._resource_request(task)[column]
            self._task_funcs[task] = task_funcs

    def _resource_request(self, task):
        if task not in self._resource_cache:
            resources = dict(RESOURCE_DEFAULTS)
            if task in self.md and not (df0 := self.md[task]).empty:
                for column in RESOURCE_DEFAULTS:
                    resources[column] = float(np.mean(df0[column]))
            self._resource_cache[task] = resources
        return self._resource_cache[task]

    def _task_instance_request(self, task, dataId):  # noqa:N803
        if task in self._task_funcs:
            tract, patch = dataId['tract'], dataId['patch']
            band = dataId.get('band', None)
            num_warps = self.num_warps(tract, patch, band=band)
            return [self._task_funcs[task][column](num_warps) for
                    column in RESOURCE_DEFAULTS]
        return tuple(self._resource_request(task).values())

    def __call__(self, job, prereq_info=None):
        cpu_time_total = 0
        memory_max = 0
        for task, count in job.quanta_counts.items():
            if task in self._task_funcs:
                dataId = job.tags  # noqa:N806
            else:
                dataId = None  # noqa:N806
            cpu_time, memory = self._task_instance_request(task, dataId)
            cpu_time_total += count*cpu_time
            memory_max = max(memory, memory_max)
        return cpu_time_total, memory_max

    def fit_resource_func(self, task, ycol, bins=20, percentile=95, deg=1):
        xcol = "num_warps"
        df0 = self.md[task]
        if not all(_ in df0 for _ in WARP_INDEX_COLS[:2]):
            raise RuntimeError(
                "Trying to fit a task without tract, patch dimensions")
        cols = WARP_INDEX_COLS
        df_warps = self.df_warps
        if WARP_INDEX_COLS[2] not in df0:  # 'band' dimension not used by task
            cols = WARP_INDEX_COLS[:2]
            df_warps = self.df_warps.groupby(cols)[xcol].sum().reset_index()
        df = df0.set_index(cols).join(df_warps.set_index(cols, drop=False),
                                      on=cols, rsuffix="_r")
        df = df.query(f"{xcol} == {xcol} and {ycol} == {ycol}")
        results = binned_statistic(
            df[xcol].to_numpy(), df[ycol].to_numpy(),
            statistic=lambda x: np.percentile(x, percentile), bins=bins)
        centers = (results.bin_edges[1:] + results.bin_edges[:-1])/2.0
        func = np.polynomial.Chebyshev.fit(centers, results.statistic, deg=deg)
        return func
