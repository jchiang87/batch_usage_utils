import pickle
import yaml
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["get_memory_factors", "plot_concurrent_jobs",
           "plot_slurm_allocation", "workflow_summary_plot"]


def get_memory_factors(memory_config_file, clustering_file,
                       mem_per_core=4000.0, use_exact_request=True):
    # Get the pipetask-specific values.
    with open(memory_config_file, "r") as fobj:
        config = yaml.safe_load(fobj)["pipetask"]
        if use_exact_request:
            memory_factors = {
                key: value["requestMemory"] / mem_per_core
                for key, value in config.items()
            }
        else:
            memory_factors = {
                key: np.ceil(value["requestMemory"] / mem_per_core)
                for key, value in config.items()
            }
    # Adjust for clustering.
    with open(clustering_file, "r") as fobj:
        config = yaml.safe_load(fobj)["cluster"]
        for cluster, spec in config.items():
            pipetasks = spec["pipetasks"].split(",")
            max_mem_factor = max([memory_factors.get(_, 1) for _ in pipetasks])
            memory_factors[cluster] = max_mem_factor
            for pipetask in pipetasks:
                memory_factors[pipetask] = max_mem_factor
    return memory_factors


def plot_concurrent_jobs(df0, label=None, alpha=1.0, scatter=True, color=None,
                         memory_factors=None, time_col_suffix="dt"):
    if memory_factors is None:
        memory_factors = {}
    times = np.concatenate((df0[f"start_{time_col_suffix}"].to_numpy(),
                            df0[f"end_{time_col_suffix}"].to_numpy()))
    deltas = np.array([memory_factors.get(task, 1.0) for task in df0["task"]])
    deltas = np.concatenate((deltas, -deltas))
    df = pd.DataFrame(dict(times=times, deltas=deltas)).sort_values("times")
    df["njobs"] = np.cumsum(df["deltas"])
    if scatter:
        plt.scatter(df["times"], df["njobs"], s=2, label=label, alpha=alpha,
                    color=color)
    else:
        plt.plot(df["times"], df["njobs"], label=label, alpha=alpha,
                 color=color)


def plot_slurm_allocation(slurm_csv_file, color=None, label=None, alpha=1.0,
                          col_substr="lsstsvc1", linestyle="-"):
    df = pd.read_csv(slurm_csv_file)
    for col in df.columns:
        if col_substr in col:
            alloc_column = col
            break
    df["time"] = [datetime.datetime.fromisoformat(_) for _ in df["Time"]]
    df["alloc_cpu"] = df[alloc_column]
    df = df.query("alloc_cpu == alloc_cpu")
    plt.plot(df["time"], df["alloc_cpu"], color=color, label=label,
             alpha=alpha, linestyle=linestyle)
    return df


def workflow_summary_plot(md_files, slurm_job_file=None,
                          htcondor_job_file=None, title=None,
                          memory_factors=None, time_col_suffix='dt',
                          show_legend=True, utc_range=None):
    dfs = []
    tasks = []
    md_all = dict()
    for md_file in md_files:
        with open(md_file, "rb") as fobj:
            md = pickle.load(fobj)
        md_all.update(md)
        for task, df0 in md.items():
            if task not in tasks:
                tasks.append(task)
            if len(df0) > 0:
                df0['task'] = [task]*len(df0)
                dfs.append(df0[["task", "start_dt", "end_dt", "start_utc",
                                "end_utc", "run_cpu_time", "run_wall_time",
                                "max_rss"]])
    df0 = pd.concat(dfs)
    if utc_range is not None:
        df0 = df0.query(f"{utc_range[0]} < start_utc < {utc_range[1]}")
        tasks = [_ for _ in tasks if _ in set(df0['task'])]
    for i, task in enumerate(tasks):
        df = df0.query(f"task == '{task}'")
        plot_concurrent_jobs(df, label=task, memory_factors=memory_factors,
                             time_col_suffix=time_col_suffix)
    plot_concurrent_jobs(df0, label="all tasks", scatter=False, color="grey",
                         alpha=0.3, memory_factors=memory_factors,
                         time_col_suffix=time_col_suffix)
    plt.xlabel("Pacific Time")
    plt.ylabel("# concurrent jobs")
    if slurm_job_file is not None:
        plot_slurm_allocation(slurm_job_file, label='slurm allocation')
    if htcondor_job_file is not None:
        plot_slurm_allocation(htcondor_job_file, label='HTCondor jobs',
                              linestyle='--', color='red')
    plt.title(title)
    if show_legend:
        plt.legend(fontsize='x-small', ncol=3, loc="upper left")
    return df0
