#!/usr/bin/env python
import pickle
import time
import argparse
from astropy.time import Time


def add_datetime(df0, mjd_col, dt_col, offset=7.):
    df0[dt_col] = Time(df0[mjd_col].to_numpy() - offset/24.,
                       format="mjd").to_datetime()
    return df0


parser = argparse.ArgumentParser()
parser.add_argument("md_file", type=str, help="pipeline metadata file")
args = parser.parse_args()

md_file = args.md_file

with open(md_file, "rb") as fobj:
    step = pickle.load(fobj)
    for task, df0 in step.items():
        print(task, end=": ")
        if len(df0) == 0:
            continue
        t0 = time.time()
        df0 = add_datetime(df0, 'start_utc', 'start_dt')
        df0 = add_datetime(df0, 'end_utc', 'end_dt')
        print(time.time() - t0)

outfile = md_file.replace("_md", "_md_dt")
with open(outfile, "wb") as fobj:
    pickle.dump(step, fobj)
