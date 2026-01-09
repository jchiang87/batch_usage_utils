import os
from collections import defaultdict
import bisect
import time
import pickle
import uuid
import numpy as np
import pandas as pd
from graphlib import TopologicalSorter

__all__ = ["JobScheduler"]


class Payload:
    def __init__(self, workflow, job_ids):
        self.job_ids = job_ids
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


def create_payloads(workflow, job_list, payload_sizes=None):
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
    payload_list = [Payload(workflow, _) for _ in payload_list]
    return payload_list


class ComputeNode:
    def __init__(self, node_id, num_cores, mem_per_core, speedup=1.0):
        self.id = node_id
        self.num_cores = num_cores
        self.mem_per_core = mem_per_core
        self.speedup = speedup
        self.occupied_cores = 0
        self.running_payloads = {}

    @property
    def free_cores(self):
        return self.num_cores - self.occupied_cores

    def run_job(self, payload, start_time):
        requested_cores = int(np.ceil(payload.memory/self.mem_per_core))
        if requested_cores > self.free_cores:
            return False
        self.occupied_cores += requested_cores
        end_time = start_time + payload.cpu_time/self.speedup
        self.running_payloads[payload.id] = (requested_cores, end_time, payload)
        return True


USDF_PARTITIONS = {
    "milano": dict(num_nodes=110,
                   cores_per_node=120,
                   mem_per_core=4.0,
                   speedup=1.0),
    "torino": dict(num_nodes=35,
                   cores_per_node=120,
                   mem_per_core=6.0,
                   speedup=2.0)
}


class ComputeCluster:
    def __init__(self, partitions=USDF_PARTITIONS):
        self.nodes = {}
        self.running_payloads = {}
        self.cores = 0
        for partition, config in partitions.items():
            for i in range(config['num_nodes']):
                node_id = f"{partition}_{i:03d}"
                self.nodes[node_id] = ComputeNode(node_id,
                                                  config['cores_per_node'],
                                                  config['mem_per_core'],
                                                  speedup=config['speedup'])
                self.cores += config['cores_per_node']
        # Maintain a list of candidate nodes for running a given job
        # in ascending order of available cores.
        self.candidate_nodes = sorted(self.nodes.values(),
                                      key=lambda x: x.free_cores)

    def run_job(self, payload, start_time, md):
        if self.candidate_nodes[-1].run_job(payload, start_time):
            # Pop the node off the list and re-insert, ordering by free cores.
            node = self.candidate_nodes.pop()
            bisect.insort(self.candidate_nodes, node,
                          key=lambda x: x.free_cores)
            self.running_payloads[(node.id, payload.id)] \
                = node.running_payloads[payload.id]
            payload.add_metadata(md, start_time)
            return True
        return False

    def delete_payload(self, key):
        node_id, payload_id = key
        cores, _, _ = self.running_payloads[key]
        del self.running_payloads[key]
        del self.nodes[node_id].running_payloads[payload_id]
        self.nodes[node_id].occupied_cores -= cores


USDF_cluster = ComputeCluster()


class JobScheduler:
    def __init__(self, compute_cluster=USDF_cluster, dt=10):
        self.ts = None  # TopologicalSorter object
        self.compute_cluster = compute_cluster
        self.dt = dt
        self.current_time = 0
        self.total_cpu_time = 0
        self.workflow = None
        self._md = defaultdict(lambda: defaultdict(list))
        self._largest_core_block \
            = compute_cluster.candidate_nodes[-1].free_cores

    def add_payload(self, payload):
        # Add random amount of sampling time, self.dt, to the start_time
        # to smooth out the scheduling.
        start_time = self.current_time + np.random.uniform(self.dt)
        if not self.compute_cluster.run_job(payload, start_time, self._md):
            return False
        return True

    def update_time(self):
        self.current_time += self.dt
        running_payloads = self.compute_cluster.running_payloads
        # Collect the finished payloads in a separate loop since they
        # can't be deleted from the running_payloads dict while iterating
        # over the items.
        finished_payloads = []
        for key, (_, end_time, _) in running_payloads.items():
            if end_time < self.current_time:
                finished_payloads.append(key)
        # Delete the finished payloads and update the cpu time total.
        for key in finished_payloads:
            cores, _, payload = running_payloads[key]
            speedup = self.compute_cluster.nodes[key[0]].speedup
            self.total_cpu_time += cores*payload.cpu_time/speedup
            payload.notify(self)
            self.compute_cluster.delete_payload(key)
        self._largest_core_block \
            = self.compute_cluster.candidate_nodes[-1].free_cores

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
        job_queue = create_payloads(self.workflow, self.ts.get_ready(),
                                    payload_sizes)
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
                if self._largest_core_block <= min_cores:
                    break
            if self.current_time % 100 == 0:
                print(self.current_time, len(job_queue),
                      len(self.compute_cluster.running_payloads), end=" ")
                if job_queue:
                    print(job_queue[-1].job_ids[0][0], flush=True)
                else:
                    print(flush=True)
            if outfile is not None and self.current_time % 10000 == 0:
                print("  Saving simulation metadata...", flush=True)
                self.save_md(outfile, clobber=True)
            new_payloads = create_payloads(self.workflow, self.ts.get_ready(),
                                           payload_sizes)
            if new_payloads:
                if shuffle:
                    job_queue.extend(new_payloads)
                    np.random.shuffle(job_queue)
                else:  # prepend new jobs to job_queue
                    new_payloads.reverse()
                    job_queue = new_payloads + job_queue
            if ((time_limit is not None and self.current_time > time_limit)
                or not self.compute_cluster.running_payloads):
                break
            self.update_time()
        print("simulated wall time:", self.current_time/3600.)
        print("total cpu time:", self.total_cpu_time/3600.)
        print("cpu time / (wall time * cores):",
              self.total_cpu_time/(self.current_time*self.compute_cluster.cores))
        print("time to run simulation:", time.time() - t0)
        if outfile is not None:
            self.save_md(outfile, clobber=True)
