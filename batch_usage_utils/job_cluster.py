from collections import defaultdict
import uuid


__all__ = ["JobCluster", "make_job_clusters"]


class JobCluster:
    def __init__(self, wf, job_ids):
        # Check that all of these jobs are for the same task.
        assert len(set(_[0] for _ in job_ids)) == 1

        self.jobs = [wf.get(_) for _ in job_ids]
        self.label = self.jobs[0].label
        self.id = uuid.uuid4()
        self.cpu_time = sum(_.cpu_time for _ in self.jobs)
        self.memory = max(_.memory for _ in self.jobs)

    def add_metadata(self, md, cluster_start_time):
        cpu_times = [_.cpu_time for _ in self.jobs]
        start_times = [cluster_start_time]
        for cpu_time in cpu_times[:-1]:
            start_times.append(start_times[-1] + cpu_time)
        for start_time, cpu_time, job in zip(start_times, cpu_times, self.jobs):
            for k, v in job.tags.items():
                md[k].append(v)
            md['start_time'].append(start_time)
            md['cpu_time'].append(cpu_time)
            md['memory'].append(self.memory)

    def notify_ts(self, ts):
        """
        Tell the scheduler's topological sorter that all of the jobs have
        finished processing.
        """
        for job in self.jobs:
            ts.done(job.id)


def make_job_clusters(job_sequence, cluster_defs=None):
    if cluster_defs is None:
        return [[_] for _ in job_sequence]

    cluster = defaultdict(list)
    job_clusters = []

    for job_id in job_sequence:
        task = job_id[0]
        if task in cluster_defs:
            cluster[task].append(job_id)
            if len(cluster[task]) >= cluster_defs[task]:
                job_clusters.append(cluster[task])
                del cluster[task]
        else:
            job_clusters.append([job_id])
    job_clusters.extend(cluster.values())
    return job_clusters
