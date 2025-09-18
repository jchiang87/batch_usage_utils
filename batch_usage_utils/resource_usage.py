from types import MappingProxyType
import numpy as np
from scipy.stats import binned_statistic
from batch_usage_utils import PipelineMetadata


__all__ = ["ResourceUsage"]


class UseVisitFunc(dict):
    def __init__(self, cpu_time, memory):
        super().__init__()
        self['run_cpu_time'] = cpu_time
        self['max_rss'] = memory


RESOURCE_DEFAULTS = MappingProxyType({"run_cpu_time": 60, "max_rss": 4.0})
RESOURCE_FIT_PERCENTILES = MappingProxyType({"run_cpu_time": 95, "max_rss": 95})

NUM_VISITS_TASKS = {
    'selectDeepCoaddVisits': UseVisitFunc(True, True),
    'selectTemplateCoaddVisits': UseVisitFunc(True, True),
    'assembleDeepCoadd': UseVisitFunc(True, True),
    'assembleTemplateCoadd': UseVisitFunc(True, True),
    'assembleCellCoadd': UseVisitFunc(True, True),
    'detectCoaddPeaks': UseVisitFunc(False, True),
    'deconvolve': UseVisitFunc(False, True),
    'deblendCoaddFootprints': UseVisitFunc(True, True),
    'measureObjectUnforced': UseVisitFunc(True, True),
    'fitDeepCoaddPsfGaussians': UseVisitFunc(True, True),
    'measureObjectForced': UseVisitFunc(True, True),
    'associateDiaSource': UseVisitFunc(True, False),
    'calculateDiaObject': UseVisitFunc(True, False),
    'standardizeObjectForcedSource': UseVisitFunc(True, False),
    'splitPrimaryObjectForcedSource': UseVisitFunc(True, False),
    'standardizeDiaObjectForcedSource': UseVisitFunc(True, False),
}


class ResourceUsage:
    def __init__(self, md_files, md_warps=None, df_warps=None):
        self.md = PipelineMetadata.read_md_files(md_files)
        self.md_warps = md_warps
        self.df_warps = df_warps
        if df_warps is not None:
            self._set_num_warps()
        # Add specialized resource request functions.
        self._task_funcs = {}
        if md_warps is not None:
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
        self._task_funcs = {}
        for task, func_status in NUM_VISITS_TASKS.items():
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

    def fit_resource_func(self, task, ycol, bins=20, percentile=95, deg=1):
        visit_index_cols = ["patch", "tract", "band"]
        xcol = "num_visits"
        df0 = self.md[task]
        if not all(_ in df0 for _ in visit_index_cols[:2]):
            raise RuntimeError(
                "Trying to fit a task without patch, tract dimensions")
        cols = visit_index_cols
        visits = self.num_visits_md
        if "band" not in df0:
            cols = visit_index_cols[:2]
            visits = self.num_visits_md.groupby(cols)[xcol].sum().reset_index()
        df = df0.set_index(cols).join(visits.set_index(cols, drop=False),
                                      on=cols, rsuffix="_r")
        # Exclude rows with nans:
        df = df.query(f"{xcol} == {xcol} and {ycol} == {ycol}")
        results = binned_statistic(
            df[xcol].to_numpy(), df[ycol].to_numpy(),
            statistic=lambda x: np.percentile(x, percentile), bins=bins)
        centers = (results.bin_edges[1:] + results.bin_edges[:-1])/2.0
        func = np.poly1d(np.polyfit(centers, results.statistic, deg))
        ymin = min(df[ycol])
        return lambda num_visits: float(max(func(num_visits), ymin))
