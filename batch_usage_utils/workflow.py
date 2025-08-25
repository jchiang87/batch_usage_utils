import os
import time
import pickle


__all__ = ["Workflow", "Job"]


class Job:
    def __init__(self, node):
        self.task_label = node.task_node.label
        self.id = node.nodeId
        self.dataId = node.quantum.dataId.to_simple().dataId
        self.inputs = set()
        self.predecessors = set()
        self.quanta_counts = {self.task_label: 1}
        self.request_memory = None

    def add_input(self, datasets):
        for ds in datasets:
            self.inputs.add(ds.id)

    @property
    def label(self):
        return self.task_label

    @property
    def tags(self):
        return self.dataId

    def __str__(self):
        return ":".join((self.task_label, str(self.dataId)))

    def __repr__(self):
        return ":".join((self.task_label, str(self.dataId)))


class Workflow(dict):
    def __init__(self, qgraph):
        super().__init__()
        dataset_map = {}

        t0 = time.time()
        for node in qgraph.graph.nodes:
            job = Job(node)
            self[job.id] = job
            for input_list in node.quantum.inputs.values():
                job.add_input(input_list)
            for output_list in node.quantum.outputs.values():
                for output in output_list:
                    dataset_map[output.id] = node.nodeId

        for job in self.values():
            for ds_id in job.inputs:
                if ds_id in dataset_map:
                    job.predecessors.add(dataset_map[ds_id])
        print("time to parse QG:", time.time() - t0)
        print(len(self))

    def get_job(self, job_name):
        return self[job_name]

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


if __name__ == '__main__':
    from load_qg import qg

    wf = Workflow(qg)
    wf.save("example_workflow.pickle", clobber=True)
