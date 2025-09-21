import os
import time
import pickle
import yaml
from tqdm import tqdm


__all__ = ["Workflow", "Job", "set_job_resource_usage"]


class Job:
    def __init__(self, node=None):
        self.inputs = set()
        self.predecessors = set()
        self._cpu_time = 90.
        self._memory = None  # if None, then use cls._memory_requests value
        self.resource_usage = None
        if node is not None:
            self.task_label = node.task_node.label
            self.id = node.nodeId
            self.dataId = node.quantum.dataId.to_simple().dataId
            self.quanta_counts = {self.task_label: 1}

    def add_input(self, datasets):
        for ds in datasets:
            self.inputs.add(ds.id)

    @property
    def label(self):
        return self.task_label

    @property
    def tags(self):
        return self.dataId

    @property
    def cpu_time(self):
        """
        Job CPU time in seconds.
        """
        return self._cpu_time

    @property
    def memory(self):
        """
        Job memory in GB.
        """
        return self._memory

    def __str__(self):
        return ": ".join((self.task_label, str(self.dataId)))

    def __repr__(self):
        return ": ".join((self.task_label, str(self.dataId)))


class Workflow(dict):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build_from_qgraph(qgraph):
        wf = Workflow()
        dataset_map = {}
        t0 = time.time()
        for node in qgraph.graph.nodes:
            job = Job(node)
            wf[job.id] = job
            for input_list in node.quantum.inputs.values():
                job.add_input(input_list)
            for output_list in node.quantum.outputs.values():
                for output in output_list:
                    dataset_map[output.id] = node.nodeId

        for job in wf.values():
            for ds_id in job.inputs:
                if ds_id in dataset_map:
                    job.predecessors.add(dataset_map[ds_id])
        print("time to parse QG:", time.time() - t0)
        print(len(wf))
        return wf

    def set_memory_requests(self, config_file, default=4.):
        """
        Read in the bps config file with the per-task memory requests and
        use those values to set the memory per job.  If there is no
        entry for a particular task, use the default, nominally 4 GB.
        """
        with open(config_file) as fobj:
            data = yaml.safe_load(fobj)
        task_memory_requests = {}
        for task, info in data['pipetask'].items():
            # Convert from MG to GB.
            task_memory_requests[task] = info['requestMemory']/1000.
        for job in tqdm(self.values()):
            job._memory = task_memory_requests.get(job.task_label, default)

    def get_job(self, job_name):
        return dict.__getitem__(self, job_name)

    def predecessors(self, job_name):
        return self[job_name].predecessors

    def save(self, outfile, clobber=False):
        if not clobber and os.path.isfile(outfile):
            raise RuntimeError(f"{outfile} exists already.")
        with open(outfile, "wb") as fobj:
            pickle.dump(self, fobj)

    @staticmethod
    def load_pickle_file(pickle_file):
        with open(pickle_file, "rb") as fobj:
            return pickle.load(fobj)


def set_job_resource_usage(workflow, resource_usage, visit_counts):
    for _, job in tqdm(workflow.items()):
        task = job.label
        dataId = job.dataId  # noqa: N806
        if "patch" not in dataId or "tract" not in dataId:
            num_visits = 1
        else:
            key = tuple(dataId[_] for _ in ["patch", "tract", "band"]
                        if _ in dataId)
            num_visits = visit_counts(*key) if len(key) >= 2 else 1
        job._cpu_time, job._memory = resource_usage(task, num_visits)
    return workflow


if __name__ == '__main__':
    from load_qg import qg

    wf = Workflow.build_from_qgraph(qg)
    wf.save("example_workflow.pickle", clobber=True)
