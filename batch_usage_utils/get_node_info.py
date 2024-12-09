import os
import glob
from collections import defaultdict
import subprocess
import pandas as pd

__all__ = ("extract_node_info",)


def extract_node_info(log_dir, clustered=False):
    command = f"grep SlotName `find {log_dir} -name \*.log -print`"
    output = subprocess.check_output(command, shell=True, encoding='utf-8')
    data = defaultdict(list)
    for line in output.split("\n"):
        if 'SlotName' not in line:
            continue
        tokens = line.split(":")
        if not clustered:
            data['detector'].append(
                int(os.path.basename(tokens[0]).split(".")[-3].split('_')[-1]))
        data['worker'].append(tokens[-1].split('@')[-1][:-len(".sdf.slac.stanford.edu")])
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    from ptc_runs import ptc_runs
    for ptc_run in ptc_runs:
        print(ptc_run, end="", flush=True)
        outfile = f"node_info_{ptc_run}_w_2024_35.parquet"
        if os.path.isfile(outfile):
            print()
            continue
        pattern = (f"/sdf/data/rubin/user/lsstccs/runs/{ptc_run}/"
                   f"submit/u/lsstccs/ptc_{ptc_run}_w_2024_35/*/jobs")
        jobs_folder = sorted(glob.glob(pattern))[-1]
        tasks = os.listdir(jobs_folder)
        if "cpPtcIsr" in tasks:
            task = "cpPtcIsr"
            clustered = False
        elif "isr_exposure" in tasks:
            task = "isr_exposure"
            clustered = True
        else:
            raise RuntimeError("no matching task")
        folders = sorted(glob.glob(os.path.join(jobs_folder, task, '*')))
        dfs = []
        print(" ", len(folders), end="")
        for i, folder in enumerate(folders):
            if i % (len(folders)//40) == 0:
                print(".", end="", flush=True)
            df = extract_node_info(folder, clustered=clustered)
            exposure = int(os.path.basename(folder))
            df['exposure'] = [exposure]*len(df)
            dfs.append(df)
        print()
        df0 = pd.concat(dfs)
        df0.to_parquet(outfile)
