import os
from collections import defaultdict
import numpy as np
import pandas as pd
from astropy.time import Time
import lsst.daf.butler as daf_butler
from lsst.obs.lsst import LsstCam


__all__ = ("extract_metadata", "add_ccd_type", "add_nproc", "add_node_info")


def extract_metadata(md, data):
    prefixes = ("prep", "init", "start", "end")
    info = md['quantum']
    # Wall times
    times = np.array([Time(info[f"{_}Utc"][:26]) for _ in prefixes])
    dt = times[1:] - times[:-1]
    for prefix, dt in zip(("prep", "init", "run"), dt):
        data[f"{prefix}_wall_time"].append(dt.sec)
    data['start_utc'].append(times[0].mjd)
    data['end_utc'].append(times[-1].mjd)
    # max RSS
    data['max_rss'].append(info['endMaxResidentSetSize']/1024**3)
    # CPU times
    times = np.array([info[f"{_}CpuTime"] for _ in prefixes])
    dt = times[1:] - times[:-1]
    for prefix, dt in zip(("prep", "init", "run"), dt):
        data[f"{prefix}_cpu_time"].append(dt)
    return data


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
    df0['nproc'] = df.query("deltas==1")['nproc']
    return df0


def add_node_info(df0, node_info):
    if 'detector' in node_info:
        node_map = dict(zip(zip(node_info['exposure'], node_info['detector']),
                            node_info['worker']))
        df0['worker'] = [node_map[key] for key in
                         zip(df0['exposure'], df0['detector'])]
    else:
        node_map = dict(zip(node_info['exposure'], node_info['worker']))
        df0['worker'] = [node_map[key] for key in df0['exposure']]
    return df0
