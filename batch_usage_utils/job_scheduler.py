import os
from collections import defaultdict
import time
import pickle
import uuid
import numpy as np
import pandas as pd
from graphlib import TopologicalSorter

__all__ = ["JobScheduler"]


class Payload:
    def __init__(self, workflow, job_ids):
        self.jobs = [workflow.get(_) for _ in job_ids]
        self.id = uuid.uuid4()
        self.cpu_time = sum(_.cpu_time for _ in self.jobs)
        self.memory = max(_.memory for _ in self.jobs)

    def add_metadata(self, md, cluster_start_time):
        cpu_times = [_.cpu_time for _ in self.jobs]
        start_times = [cluster_start_time]
        for cpu_time in cpu_times[:-1]:
            start_times.append(start_times[-1] + cpu_time)
        for start_time, cpu_time, job in zip(start_times, cpu_times, self.jobs):
            task = job.label
            for k, v in job.tags.items():
                md[task][k].append(v)
            md[task]['start_time'].append(start_time)
            md[task]['cpu_time'].append(cpu_time)
            md[task]['memory'].append(self.memory)

    def notify(self, scheduler):
        """
        Tell the scheduler that all of the jobs have finished processing.
        """
        for job in self.jobs:
            scheduler.done(job.id)


def create_payloads(job_list, payload_sizes=None):
    if payload_sizes is None:
        return [[_] for _ in job_list]

    jobs = defaultdict(list)
    payload_list = []
    for job_id in job_list:
        task = job_id[0]
        if task in payload_sizes:
            jobs[task].append(job_id)
            if len(jobs[task]) >= payload_sizes[task]:
                payload_list.append(jobs[task])
                del jobs[task]
        else:
            payload_list.append([job_id])
    payload_list.extend(jobs.values())
    return payload_list


class JobScheduler:
    def __init__(self, nodes=10, cores_per_node=120, mem_per_core=4, dt=10):
        self.ts = None
        self.cores = nodes*cores_per_node
        self.available_cores = self.cores
        self.mem_per_core = mem_per_core
        self.dt = dt
        self.running_payloads = {}
        self.current_time = 0
        self.total_cpu_time = 0
        self.workflow = None
        self._md = defaultdict(lambda: defaultdict(list))

    def add_payload(self, job_ids):
        payload = Payload(self.workflow, job_ids)
        requested_cores = max(1, int(np.ceil(payload.memory/self.mem_per_core)))
        if self.available_cores < requested_cores:
            return False
        # Add random amount of sampling time, self.dt, to the start_time
        # to smooth out the scheduling.
        start_time = self.current_time + np.random.uniform(self.dt)
        end_time = start_time + payload.cpu_time
        self.running_payloads[payload.id] = (requested_cores, end_time, payload)
        payload.add_metadata(self._md, start_time)
        self.available_cores -= requested_cores
        return True

    def update_time(self):
        self.current_time += self.dt
        self._compute_available_cores()
        if sum(_[0] for _ in self.running_payloads.values()) > self.cores:
            raise RuntimeError("Too many running payloads.")

    def _compute_available_cores(self):
        occupied_cores = 0
        finished_payloads = []
        for payload_id, (cores, end_time, _) in self.running_payloads.items():
            if end_time < self.current_time:
                finished_payloads.append(payload_id)
            else:
                occupied_cores += cores
        for payload_id in finished_payloads:
            cores, _, payload = self.running_payloads[payload_id]
            self.total_cpu_time += cores*payload.cpu_time
            payload.notify(self)
            del self.running_payloads[payload_id]
        self.available_cores = self.cores - occupied_cores

    def done(self, job_id):
        self.ts.done(job_id)

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
               min_cores=5, payload_sizes=None):
        t0 = time.time()
        self.workflow = workflow
        self.ts = TopologicalSorter()
        for job_id in workflow:
            predecessors = list(workflow.predecessors(job_id))
            self.ts.add(job_id, *predecessors)
        self.ts.prepare()
        print("TopologicalSorter prep time:", time.time() - t0)

        t0 = time.time()
        job_queue = create_payloads(self.ts.get_ready(), payload_sizes)
        if shuffle:
            np.random.shuffle(job_queue)
        else:
            # Process jobs in returned order, reversing the list so
            # that jobs can be popped off the end of the list without
            # invalidating lower-valued indexes.
            job_queue.reverse()
        while self.ts.is_active():
            # Iterate from the end of job_queue so that successfully
            # scheduled jobs can be popped off the end of the list.
            for i in range(len(job_queue) - 1, -1, -1):
                if self.add_payload(job_queue[i]):
                    job_queue.pop(i)
                if self.available_cores <= min_cores:
                    break
            if self.current_time % 100 == 0:
                print(self.current_time, len(job_queue),
                      len(self.running_payloads), end=" ")
                if job_queue:
                    print(job_queue[-1][0][0], flush=True)
                else:
                    print(flush=True)
            if outfile is not None and self.current_time % 10000 == 0:
                print("  Saving simulation metadata...", flush=True)
                self.save_md(outfile, clobber=True)
            new_payloads = create_payloads(self.ts.get_ready(), payload_sizes)
            if new_payloads:
                if shuffle:
                    job_queue.extend(new_payloads)
                    np.random.shuffle(job_queue)
                else:  # prepend new jobs to job_queue
                    new_payloads.reverse()
                    job_queue = new_payloads + job_queue
            if ((time_limit is not None and self.current_time > time_limit)
                or not self.running_payloads):
                break
            self.update_time()
        print("simulated wall time:", self.current_time/3600.)
        print("total cpu time:", self.total_cpu_time/3600.)
        print("cpu time / (wall time * cores):",
              self.total_cpu_time/(self.current_time*self.cores))
        print("time to run simulation:", time.time() - t0)
        if outfile is not None:
            self.save_md(outfile, clobber=True)
