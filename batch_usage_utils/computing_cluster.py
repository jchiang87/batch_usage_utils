from collections import defaultdict
import time
import numpy as np
from graphlib import TopologicalSorter

__all__ = ["ComputingCluster"]


class ComputingCluster:
    def __init__(self, resource_usage, nodes=10, cores_per_node=120,
                 mem_per_core=4, dt=10):
        self.ts = None
        self.resource_usage = resource_usage
        self.cores = nodes*cores_per_node
        self.available_cores = self.cores
        self.mem_per_core = mem_per_core
        self.dt = dt
        self.running_jobs = {}
        self.current_time = 0
        self.total_cpu_time = 0
        self.gwf = None
        self.md = defaultdict(lambda : defaultdict(list))

    def add_job(self, job):
        gwf_job = self.gwf.get_job(job)
        task = gwf_job.label
        cpu_time, memory = self.resource_usage(gwf_job)
        requested_cores = max(1, int(np.ceil(memory/self.mem_per_core)))
        if self.available_cores < requested_cores:
            return False
        self.running_jobs[job] = (requested_cores,
                                  self.current_time + cpu_time,
                                  cpu_time)
        for k, v in gwf_job.tags.items():
            self.md[task][k].append(v)
        self.md[task]['start_time'].append(self.current_time)
        self.md[task]['cpu_time'].append(cpu_time)
        self.md[task]['memory'].append(memory)
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

    def submit(self, gwf):
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
        # Repackage the metadata as data frames
        for key, data in self.md.items():
            self.md[key] = pd.DataFrame(data)
        print("simulated wall time:", self.current_time/3600.)
        print("total cpu time:", self.total_cpu_time/3600.)
        print("time to run simulation:", time.time() - t0)
