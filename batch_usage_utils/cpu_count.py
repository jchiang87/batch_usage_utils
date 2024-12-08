import os
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta


def utc_offset(pacific_time):
    """Return the astropy.time.TimeDelta offset to add to
       Pacific time to convert to UTC."""
    # Pacific times for DST start and end.
    dst_start_mjd = Time("2024-03-10 02:00:00")
    dst_end_mjd = Time("2024-11-03T02:00:00")
    if (pacific_time < dst_start_mjd or
        pacific_time > dst_end_mjd):
        return TimeDelta(8.0/24.0, format='jd')
    return TimeDelta(7.0/24.0, format='jd')


class CpuCount:
    def __init__(self, sacct_info_file):
        with open(sacct_info_file) as fobj:
            lines = fobj.readlines()
        data = defaultdict(list)
        for line in lines:
            if "glide_" not in line:
                continue
            tokens = line.split()
            data['job_id'].append(tokens[0])
            data['partition'].append(tokens[2])
            data['alloc_cpus'].append(int(tokens[3]))
            elapsed_time = self.parse_elapsed_time(tokens[4])
            data['elapsed_time'].append(elapsed_time)
            start_time = Time(tokens[7])
            offset = utc_offset(start_time)
            data['start_time'].append(start_time + offset)
            start_mjd = (start_time + offset).mjd
            end_mjd = start_mjd + elapsed_time/86400.
            data['start_mjd'].append(start_mjd)
            data['end_mjd'].append(end_mjd)
            data['node'].append(tokens[8])
        self.df0 = pd.DataFrame(data)
        self._setup_arrays()

    def _setup_arrays(self):
        times = sorted(np.concatenate((self.df0['start_mjd'].to_numpy(),
                                       self.df0['end_mjd'].to_numpy())))
        times = np.unique(times)
        num_cpus = [sum(self.df0.query(f"start_mjd <= {_} <= end_mjd")
                        ['alloc_cpus']) for _ in times]

        self.times = [times[0]]
        self.num_cpus = [num_cpus[0]]
        for i in range(1, len(times)):
            self.times.append(times[i])
            self.num_cpus.append(num_cpus[i-1])
            self.times.append(times[i])
            self.num_cpus.append(num_cpus[i])

    @staticmethod
    def make_step_function(xvals, yvals):
        xx = [xvals[0]]
        yy = [yvals[0]]
        for i in range(1, len(xvals)):
            xx.extend((xvals[i], xvals[i]))
            yy.extend((yvals[i-1], yvals[i]))
        return xx, yy

    def interp(self, mjds):
        return np.interp(mjds, self.times, self.num_cpus)

    def __call__(self, mjd):
        df = self.df0.query(f"start_mjd <= {mjd} <= end_mjd")
        return sum(df['alloc_cpus'])

    def add_column(self, df):
        df['cpu_count'] = self.interp(df['start_utc'].to_numpy())
        return df

    @staticmethod
    def parse_elapsed_time(entry):
        tokens = entry.split(':')
        elapsed = (int(tokens[0])*60. + int(tokens[1]))*60. + int(tokens[2])
        return elapsed

    @staticmethod
    def read_file(infile):
        with open(infile, "rb") as fobj:
            return pickle.load(fobj)

    def write_file(self, outfile, clobber=False):
        if not clobber and os.path.isfile(outfile):
            raise RuntimeError(f"File {outfile} exists.")
        with open(outfile, "wb") as fobj:
            pickle.dump(self, fobj)

    def plot_node_history(self, mjd_min, mjd_max):
        df0 = self.df0.query(f"{mjd_min} <= end_mjd and start_mjd <= {mjd_max}")
        nodes = sorted(set(df0['node']))
        for lw, node in enumerate(nodes):
            df = pd.DataFrame(df0.query(f"node=='{node}'"))
            times = np.concatenate((df['start_mjd'], df['end_mjd']))
            deltas = np.concatenate((df['alloc_cpus'], -df['alloc_cpus']))
            my_df = pd.DataFrame(dict(times=times, deltas=deltas))\
                      .sort_values('times')
            my_df['ncpus'] = np.cumsum(my_df['deltas'])
            times, ncpus = self.make_step_function(my_df['times'].to_numpy(),
                                                   my_df['ncpus'].to_numpy())
            plt.plot(times, ncpus, linewidth=(len(nodes)-lw), label=node)
        plt.legend(fontsize='x-small')
        plt.xlim(mjd_min, mjd_max)


if __name__ == '__main__':
    import time
#    sacct_info_file = "sacct_output.txt"
    sacct_info_file = "sacct_jchiang_output_2024-12-01_2024-12-06.txt"
    t0 = time.time()
    cpu_count = CpuCount(sacct_info_file)
    t1 = time.time()
    print(t1 - t0, flush=True)
    with open("cpu_count_jchiang.pickle", "wb") as fobj:
        pickle.dump(cpu_count, fobj)

#    df0 = pd.read_parquet("cpPtcIsr_metadata_E1145_md_stats.parquet")
#    t2 = time.time()
#    print(t2 - t1, flush=True)
#
#    df0 = cpu_count.add_column(df0)
#    t3 = time.time()
#    print(t3 - t2, flush=True)
#    df0.to_parquet("foo.parquet")
