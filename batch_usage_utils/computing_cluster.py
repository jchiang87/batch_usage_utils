from collections import defaultdict
import time
import numpy as np
import pandas as pd
from graphlib import TopologicalSorter

__all__ = ["ComputingCluster"]


class ComputingCluster:
    def __init__(self, resource_usage, nodes=10, cores_per_node=120,
                 mem_per_core=4, dt=10, use_requestedMemory=True):
        self.ts = None
        self.resource_usage = resource_usage
        self.cores = nodes*cores_per_node
        self.available_cores = self.cores
        self.mem_per_core = mem_per_core
        self.dt = dt
        self.use_requestedMemory = use_requestedMemory
        self.running_jobs = {}
        self.current_time = 0
        self.total_cpu_time = 0
        self.gwf = None
        self._md = defaultdict(lambda : defaultdict(list))

    def add_job(self, job):
        gwf_job = self.gwf.get_job(job)
        task = gwf_job.label
        cpu_time, memory = self.resource_usage(gwf_job)
        if self.use_requestedMemory:
            memory = gwf_job.request_memory/1024.0
        requested_cores = max(1, int(np.ceil(memory/self.mem_per_core)))
        if self.available_cores < requested_cores:
            return False
        self.running_jobs[job] = (requested_cores,
                                  self.current_time + cpu_time,
                                  cpu_time)
        for k, v in gwf_job.tags.items():
            self._md[task][k].append(v)
        self._md[task]['start_time'].append(self.current_time)
        self._md[task]['cpu_time'].append(cpu_time)
        self._md[task]['memory'].append(memory)
        self.available_cores -= requested_cores
        return True

    def update_time(self):
        self.current_time += self.dt
        self._compute_available_cores()
        if sum(_[0] for _ in self.running_jobs.values()) > self.cores:
            raise RuntimeError("too many running jobs")

    @property
    def occupied_cores(self):
        return sum(_[0] for _ in self.running_jobs.values())

    def _compute_available_cores(self):
        occupied_cores = 0
        finished_jobs = []
        for job, (cores, end_time, cpu_time) in self.running_jobs.items():
            if end_time < self.current_time:
                finished_jobs.append(job)
            else:
                occupied_cores += cores
        for job in finished_jobs:
            cores, _, cpu_time = self.running_jobs[job]
            self.total_cpu_time += cores*cpu_time
            del self.running_jobs[job]
            self.ts.done(job)
        self.available_cores = self.cores - occupied_cores

    @property
    def md(self):
        # Repackage the accumulated task metadata as a dict of data frames.
        return {key: pd.DataFrame(data) for key, data in self._md.items()}

    def save_md(self, outfile, clobber=False):
        if not clobber and os.path.isfile(outfile):
            raise FileExistsError(f"{outfile} exists already")
        with open(outfile, "wb") as fobj:
            pickle.dump(self.md, fobj)

    def submit(self, gwf, time_limit=None):
        self.gwf = gwf
        self.ts = TopologicalSorter()
        for job_name in gwf:
            predecessors = list(gwf.predecessors(job_name))
            self.ts.add(job_name, *predecessors)
        self.ts.prepare()
        t0 = time.time()

        job_sequence = list(self.ts.get_ready())
        while self.ts.is_active() or self.running_jobs:
            if ncores := self.available_cores > 0 and job_sequence:
                indexes = list(range(len(job_sequence)))
                indexes.reverse()
                added_jobs = 0
                for i in indexes:
                    if self.add_job(job_sequence[i]):
                        job_sequence.pop(i)
                        added_jobs += 1
                    if added_jobs == ncores:
                        break
            self.update_time()
            if self.current_time % 100 == 0:
                print(self.current_time, self.occupied_cores, end="  ")
                print(len([_ for _ in self.running_jobs.keys()
                           if gwf.get_job(_).label == 'coadd']))
            job_sequence.extend(self.ts.get_ready())
            if time_limit is not None and self.current_time > time_limit:
                break
        print("simulated wall time:", self.current_time/3600.)
        print("total cpu time:", self.total_cpu_time/3600.)
        print("cpu time / (wall time * cores):",
              self.total_cpu_time/(self.current_time*self.cores))
        print("time to run simulation:", time.time() - t0)
