import os
from collections import defaultdict
import time
import pickle
import numpy as np
import pandas as pd
from graphlib import TopologicalSorter

__all__ = ["ComputingCluster"]


class ComputingCluster:
    def __init__(self, nodes=10, cores_per_node=120, mem_per_core=4, dt=10):
        # TODO: add back use_requestedMemory option
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

    def add_job(self, job_name):
        job = self.workflow.get_job(job_name)
        task = job.label
        cpu_time, memory = job.cpu_time, job.memory
        # add random amount of sampling time to the start_time to smooth
        # out the scheduling
        start_time = self.current_time + np.random.uniform(self.dt)
        requested_cores = max(1, int(np.ceil(memory/self.mem_per_core)))
        if self.available_cores < requested_cores:
            return False
        self.running_jobs[job_name] = (requested_cores,
                                       start_time + cpu_time,
                                       cpu_time)
        for k, v in job.tags.items():
            self._md[task][k].append(v)
        self._md[task]['start_time'].append(start_time)
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

    def submit(self, workflow, time_limit=None, outfile=None, shuffle=False):
        t0 = time.time()
        self.workflow = workflow
        self.ts = TopologicalSorter()
        for job_name in workflow:
            predecessors = list(workflow.predecessors(job_name))
            self.ts.add(job_name, *predecessors)
        self.ts.prepare()
        print("TopologicalSorter prep time:", time.time() - t0)

        t0 = time.time()
        job_sequence = list(self.ts.get_ready())
        if shuffle:
            np.random.shuffle(job_sequence)
        else:
            # Process jobs in returned order, reversing the list so
            # that jobs can be popped off the end of the list without
            # invalidating lower-valued indexes.
            job_sequence.reverse()
        while self.ts.is_active() or self.running_jobs:
            if self.available_cores > 0 and job_sequence:
                indexes = list(range(len(job_sequence)))
                indexes.reverse()  # reverse to enable popping off end of list
                for i in indexes:
                    if self.add_job(job_sequence[i]):
                        job_sequence.pop(i)
                    if self.available_cores == 0:
                        break
            if self.current_time % 100 == 0:
                print(self.current_time, len(job_sequence), flush=True)
            if outfile is not None and self.current_time % 10000 == 0:
                print("  saving simulation metadata", flush=True)
                self.save_md(outfile, clobber=True)
            new_jobs = list(self.ts.get_ready())
            if new_jobs:
                if shuffle:
                    job_sequence.extend(new_jobs)
                    np.random.shuffle(job_sequence)
                else:  # prepend new jobs to job_sequence
                    new_jobs.reverse()
                    job_sequence = new_jobs + job_sequence
            if time_limit is not None and self.current_time > time_limit:
                break
            self.update_time()
        print("simulated wall time:", self.current_time/3600.)
        print("total cpu time:", self.total_cpu_time/3600.)
        print("cpu time / (wall time * cores):",
              self.total_cpu_time/(self.current_time*self.cores))
        print("time to run simulation:", time.time() - t0)
