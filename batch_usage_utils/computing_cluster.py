import os
from collections import defaultdict
import time
import pickle
import numpy as np
import pandas as pd
from graphlib import TopologicalSorter
from .job_cluster import JobCluster, make_job_clusters

__all__ = ["ComputingCluster"]


class ComputingCluster:
    def __init__(self, nodes=10, cores_per_node=120, mem_per_core=4, dt=10):
        self.ts = None
        self.cores = nodes*cores_per_node
        self.available_cores = self.cores
        self.mem_per_core = mem_per_core
        self.dt = dt
        self.running_jobs = {}
        self.current_time = 0
        self.total_cpu_time = 0
        self.workflow = None
        self._md = defaultdict(lambda: defaultdict(list))

    def add_jobs(self, job_names):
        job = JobCluster(self.workflow, job_names)
        job_name = job.id
        cpu_time, memory = job.cpu_time, job.memory
        # add random amount of sampling time to the start_time to smooth
        # out the scheduling
        start_time = self.current_time + np.random.uniform(self.dt)
        requested_cores = max(1, int(np.ceil(memory/self.mem_per_core)))
        if self.available_cores < requested_cores:
            return False
        self.running_jobs[job_name] = (requested_cores,
                                       start_time + cpu_time,
                                       cpu_time,
                                       job)
        job.add_metadata(self._md, start_time)
        self.available_cores -= requested_cores
        return True

    def update_time(self):
        self.current_time += self.dt
        self._compute_available_cores()
        if sum(_[0] for _ in self.running_jobs.values()) > self.cores:
            raise RuntimeError("too many running jobs")

    def _compute_available_cores(self):
        occupied_cores = 0
        finished_jobs = []
        for job, (cores, end_time, cpu_time, _) in self.running_jobs.items():
            if end_time < self.current_time:
                finished_jobs.append(job)
            else:
                occupied_cores += cores
        for job in finished_jobs:
            cores, _, cpu_time, job_instance = self.running_jobs[job]
            self.total_cpu_time += cores*cpu_time
            job_instance.notify_ts(self.ts)
            del self.running_jobs[job]
        self.available_cores = self.cores - occupied_cores

    @property
    def md(self):
        # Repackage the accumulated task metadata as a dict of data frames.
        return {key: pd.DataFrame(data) for key, data in self._md.items()}

    def save_md(self, outfile, clobber=False):
        if not clobber and os.path.isfile(outfile):
            raise FileExistsError(f"{outfile} exists already")
        md = self.md
        with open(outfile, "wb") as fobj:
            pickle.dump(md, fobj)

    def submit(self, workflow, time_limit=None, outfile=None, shuffle=False,
               min_cores=5, cluster_defs=None):
        t0 = time.time()
        self.workflow = workflow
        self.ts = TopologicalSorter()
        for job_name in workflow:
            predecessors = list(workflow.predecessors(job_name))
            self.ts.add(job_name, *predecessors)
        self.ts.prepare()
        print("TopologicalSorter prep time:", time.time() - t0)

        t0 = time.time()
        job_sequence = make_job_clusters(self.ts.get_ready(), cluster_defs)
        if shuffle:
            np.random.shuffle(job_sequence)
        else:
            # Process jobs in returned order, reversing the list so
            # that jobs can be popped off the end of the list without
            # invalidating lower-valued indexes.
            job_sequence.reverse()
        while self.ts.is_active():
            # Iterate from the end of job_sequence so that successfully
            # scheduled jobs can be popped off the end of the list.
            for i in range(len(job_sequence) - 1, -1, -1):
                if self.add_jobs(job_sequence[i]):
                    job_sequence.pop(i)
                if self.available_cores <= min_cores:
                    break
            if self.current_time % 100 == 0:
                print(self.current_time, len(job_sequence), flush=True)
            if outfile is not None and self.current_time % 10000 == 0:
                print("  saving simulation metadata", flush=True)
                self.save_md(outfile, clobber=True)
            new_jobs = make_job_clusters(self.ts.get_ready(), cluster_defs)
            if new_jobs:
                if shuffle:
                    job_sequence.extend(new_jobs)
                    np.random.shuffle(job_sequence)
                else:  # prepend new jobs to job_sequence
                    new_jobs.reverse()
                    job_sequence = new_jobs + job_sequence
            if ((time_limit is not None and self.current_time > time_limit)
                or not self.running_jobs):
                break
            self.update_time()
        print("simulated wall time:", self.current_time/3600.)
        print("total cpu time:", self.total_cpu_time/3600.)
        print("cpu time / (wall time * cores):",
              self.total_cpu_time/(self.current_time*self.cores))
        print("time to run simulation:", time.time() - t0)
        if outfile is not None:
            self.save_md(outfile, clobber=True)
