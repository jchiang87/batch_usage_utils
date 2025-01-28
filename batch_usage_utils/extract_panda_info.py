"""Module to extract job info from PanDA workflows."""

from collections import defaultdict
import time
import pandas as pd
from pandaclient import Client
from lsst.ctrl.bps import BPS_DEFAULTS, BpsConfig
from lsst.ctrl.bps.panda.utils import get_idds_client

__all__ = ["PandaJobInfo"]


class PandaJobInfo:
    """Class to extract job information using the PanDA python client"""

    def __init__(self):
        config = BpsConfig(BPS_DEFAULTS)
        self.idds_client = get_idds_client(config)

    def get_job_info(self, workflow_ids, outfile=None, verbose=False):
        """Get job info for a list of workflow_ids.

        Paramters
        ---------
        workflow_ids : list
            List of PanDA workflow IDs.
        outfile : str [None]
            Name of parquet file to write out the data frame.  If None,
            then don't write a file.
        verbose : bool [False]
            Verbosity flag.

        Returns
        -------
        pandas.DataFrame
        """
        t0 = time.time()
        dfs = []
        for workflow_id in workflow_ids:
            print(workflow_id, (time.time() - t0) / 60.0, flush=True)
            results = self.idds_client.get_requests(
                request_id=workflow_id, with_detail=True
            )
            task_ids = [_["transform_workload_id"] for _ in results[1][1]]
            for task_id in task_ids:
                _, panda_ids = Client.getPandaIDsWithTaskID(task_id)
                df = self._fill_job_info_dataframe(panda_ids, verbose)
                df["workflow_id"] = len(df) * [workflow_id]
                dfs.append(df)
        df0 = pd.concat(dfs)
        if outfile is not None:
            df0.to_parquet(outfile)
        return df0

    def _fill_job_info_dataframe(self, panda_ids, verbose):
        status, results = Client.getFullJobStatus(ids=panda_ids, verbose=verbose)
        if status != 0:
            raise RuntimeError(
                "pandaclient.Client.getFullJobStatus:\n"
                f"non-zero return status: {status}"
            )
        data = defaultdict(list)
        for result in results:
            if result is None:
                continue
            job_info = result.to_dict()
            job_name = job_info["jobName"]
            if "pipetaskInit" in job_name or "finalJob" in job_name:
                continue
            uuids = (
                job_info["jobParameters"]
                .split("qgraphNodeId:")[1]
                .split("+qgraphId")[0]
                .split(",")
            )
            for uuid in uuids:
                data["uuid"].append(uuid)
                data["worker_node"].append(job_info["modificationHost"])
                data["panda_id"].append(job_info["PandaID"])
                data["job_name"].append(job_name)
                data["task_id"].append(job_info["taskID"])
                data["jobset_id"].append(job_info["jobsetID"])
                data["start_time"].append(job_info["startTime"])
                data["end_time"].append(job_info["endTime"])
        return pd.DataFrame(data)
