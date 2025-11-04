from types import MappingProxyType
import matplotlib.pyplot as plt
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
    def __init__(self, md_files, num_visits_md):
        self.md = PipelineMetadata.read_md_files(md_files)
        self.num_visits_md = num_visits_md
        # Add specialized resource request functions.
        self._add_task_funcs()
        self._resource_cache = {}

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
                        = lambda x: self._fixed_resource_request(task)[column]
            self._task_funcs[task] = task_funcs

    def __call__(self, task, num_visits):  # noqa:N803
        if task in self._task_funcs:
            return [self._task_funcs[task][column](num_visits) for
                    column in RESOURCE_DEFAULTS]
        return tuple(self._fixed_resource_request(task).values())

    def _fixed_resource_request(self, task):
        if task not in self._resource_cache:
            resources = dict(RESOURCE_DEFAULTS)
            if task in self.md and not (df0 := self.md[task]).empty:
                for column in RESOURCE_DEFAULTS:
                    resources[column] = float(np.mean(df0[column]))
            self._resource_cache[task] = resources
        return self._resource_cache[task]

    def fit_resource_func(self, task, ycol, bins=20, percentile=95, deg=1,
                          make_plot=False):
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
        if make_plot:
            plt.scatter(df[xcol], df[ycol], s=2)
            xmin, xmax, _, _ = plt.axis()
            xx = np.linspace(xmin, xmax, 100)
            yy = func(xx)
            plt.plot(xx, yy, linestyle=":", color='green')
            plt.axvline(0, color='red', linestyle='--')
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel(xcol)
            plt.ylabel(ycol)
            plt.title(task)
        return lambda num_visits: float(max(func(num_visits), ymin))
