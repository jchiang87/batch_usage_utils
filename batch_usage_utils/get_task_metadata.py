import os
import glob
from collections import defaultdict
import json
import yaml
import numpy as np
import pandas as pd
from astropy.time import Time
import lsst.daf.butler as daf_butler
from lsst.obs.lsst import LsstCam


__all__ = ("extract_metadata", "extract_md_json", "extract_md_yaml",
           "add_ccd_type", "add_nproc", "add_nproc_avg", "add_node_info",
           "get_task_names")


def extract_metadata(md, data):
    prefixes = ("prep", "init", "start", "end")
    info = md['quantum']
    # Wall times
    times = np.array([Time(info[f"{_}Utc"].split('+')[0]) for _ in ("prep", "end")])
    dt = times[1:] - times[:-1]
    for prefix, dt in zip(("run",), dt):
        data[f"{prefix}_wall_time"].append(dt.sec)
    data['start_utc'].append(times[0].mjd)
    data['end_utc'].append(times[-1].mjd)
    # max RSS
    data['max_rss'].append(info['endMaxResidentSetSize']/1024**3)
    # major page faults
    data['major_page_faults'].append(info['endMajorPageFaults'])
    # CPU times
    times = []
    for prefix in prefixes:
        try:
            value = info[f"{prefix}CpuTime"]
        except KeyError:
            value = info[f"endCpuTime"]
        times.append(value)
    times = np.array(times)
    dt = times[1:] - times[:-1]
    for prefix, dt in zip(("prep", "init", "run"), dt):
        data[f"{prefix}_cpu_time"].append(dt)
    # Node info
    if 'nodeName' in info:
        data['node_name'].append(info['nodeName'])
    # Butler put/get info
    if 'butler_metrics' in info:
        for key in ('time_in_put', 'time_in_get', 'n_get', 'n_put'):
            data[key].append(info['butler_metrics'][key])
    else:
        for key in ('time_in_put', 'time_in_get', 'n_get', 'n_put'):
            data[key].append(None)
    return data


def extract_md_json(json_file, data):
    with open(json_file) as fobj:
        json_obj = json.load(fobj)
    info = {}
    md = {'quantum': info}
    for key, value in json_obj['metadata']['quantum']['scalars'].items():
        info[key] = value
    for key, value in json_obj['metadata']['quantum']['arrays'].items():
        try:
            info[key] = value[0]
        except IndexError as eobj:
            pass
    if 'butler_metrics' not in json_obj['metadata']['quantum']['metadata']:
        # Do nothing by raising FileNotFoundError to avoid adding dataId info
        #raise FileNotFoundError("no butler metrics")
        pass
    else:
        info['butler_metrics'] = {}
        for key, value in json_obj['metadata']['quantum']['metadata']\
            ['butler_metrics']['scalars'].items():
            try:
                info['butler_metrics'][key] = value
            except IndexError as eobj:
                pass
    return extract_metadata(md, data)


def extract_md_yaml(yaml_file, data):
    with open(yaml_file) as fobj:
        yaml_obj = yaml.safe_load(fobj)
    md = {'quantum': yaml_obj.get_dict("quantum")}
    return extract_metadata(md, data)


camera = LsstCam.getCamera()
ccd_type_dict = {det.getId(): det.getPhysicalType() for det in camera}
def add_ccd_type(df0):
    df0['ccd_type'] = [ccd_type_dict[_] for _ in df0['detector']]
    return df0


def add_nproc(df0):
    """Add column with number of running processes at
    each start_utc time."""
    times = np.concatenate((df0['start_utc'], df0['end_utc']))
    deltas = np.concatenate((np.ones(len(df0)), -np.ones(len(df0))))
    df = pd.DataFrame(dict(times=times, deltas=deltas)).sort_values('times')
    df['nproc'] = np.cumsum(df['deltas'])
    df0['nproc'] = df.query("deltas==1")['nproc'].to_list()
    return df0


class Nproc:
    def __init__(self, df0):
        times = np.concatenate((df0['start_utc'], df0['end_utc']))
        deltas = np.concatenate((np.ones(len(df0)), -np.ones(len(df0))))
        self.df = pd.DataFrame(dict(times=times, deltas=deltas)).sort_values('times')
        self.df['nproc'] = np.cumsum(self.df['deltas'])

    def __call__(self, times):
        return np.interp(times, self.df['times'], self.df['nproc'])


def add_nproc_avg(df0, nsamp=10):
    """Add a column with the number of concurrent processes on the
    associated node averaged over the start and stop times of the
    process at issue.
    """
    nodes = sorted(set(df0['node_name']))
    nproc_map = {}
    for node_name in nodes:
        nproc_map[node_name] = Nproc(pd.DataFrame(df0.query(f"node_name=='{node_name}'")))
    nproc = []
    for _, row in tqdm(df0.iterrows()):
        times = np.linspace(row.start_utc, row.end_utc, nsamp)
        nproc.append(np.mean(nproc_map[row.node_name](times)))
    df0['nproc'] = nproc
    return df0


def add_node_info(df0, node_info):
    if 'detector' in node_info and 'exposure' in node_info:
        node_map = dict(zip(zip(node_info['exposure'], node_info['detector']),
                            node_info['worker']))
        df0['worker'] = [node_map[key] for key in
                         zip(df0['exposure'], df0['detector'])]
    elif 'exposure' in node_info:
        node_map = dict(zip(node_info['exposure'], node_info['worker']))
        df0['worker'] = [node_map[key] for key in df0['exposure']]
    elif 'detector' in node_info:
        node_map = dict(zip(node_info['detector'], node_info['worker']))
        df0['worker'] = [node_map[key] for key in df0['detector']]
    return df0


def get_task_names(submit_dir_root, pattern=None):
    if pattern is None:
        pattern = f"{submit_dir_root}/*/quantumGraph*.out"
    log_files = sorted(glob.glob(pattern))
    tasks = {}
    for log_file in log_files:
        with open(log_file) as fobj:
            lines = fobj.readlines()
        line_types = [_[:len("Quanta")] for _ in lines]
        try:
            index0 = line_types.index("Quanta") + 2
        except ValueError:
            continue
        index1 = line_types[index0:].index("VERBOS") + index0
        for line in lines[index0:index1]:
            task = line.strip().split()[-1]
            count = int(line.strip().split()[0])
            if task not in tasks:
                tasks[task] = count
    return tasks
