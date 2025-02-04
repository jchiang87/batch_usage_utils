import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ("plot_batch_stats",)


def plot_batch_stats(df0, figsize=(10, 15), title=None):
    plt.figure(figsize=figsize)
    ny, nx = 3, 2

    start_utc = df0['start_utc'].to_numpy()
    dt = np.concatenate(([0], start_utc[1:] - start_utc[:-1]))
    mjd0 = int(min(start_utc))
    start_utc = 24.*(start_utc - mjd0)

    plt.subplot(ny, nx, 1)
    plt.scatter(start_utc, df0['cpu_count'], s=2, label='alloc_cpus')
    plt.scatter(start_utc, df0['nproc'], s=2, label='nproc')
    plt.xlabel(f'start_utc (24*(MJD - {mjd0}))')
    plt.ylabel('# instances')
    plt.legend(fontsize='x-small')

    plt.subplot(ny, nx, 2)
    plt.scatter(start_utc, np.cumsum(dt*df0['cpu_count']), s=2,
                label='alloc_cpus*dt')
#    dt_nproc = (df0['end_utc'].to_numpy() - start_utc)
#    plt.scatter(start_utc, np.cumsum(dt_nproc*df0['nproc']), s=2,
#                label='nproc*dt')
    plt.xlabel(f'start_utc (24*(MJD - {mjd0}))')
    plt.ylabel('# instances * time')
    plt.legend(fontsize='x-small')

    ccd_types = sorted(set(df0['ccd_type']))
    workers = sorted(set(df0['worker']))
    plt.subplot(ny, nx, 3)
    for worker in workers:
        df = df0.query(f"worker == '{worker}'")
        plt.scatter(24*(df['start_utc'] - mjd0), df['run_wall_time'], s=2,
                    label=worker)
    plt.legend(fontsize='x-small')
    plt.xlabel(f'start_utc (24*(MJD - {mjd0}))')
    plt.ylabel('run_wall_time (s)')
    cpu_counts = sorted(set(df0['cpu_count']))

    plt.subplot(ny, nx, 4)
    column = "run_wall_time"
    for count in cpu_counts:
        df = df0.query(f"cpu_count == {count}")
        plt.hist(np.log10(df[column]), bins=100, label=str(int(count)),
                 alpha=0.5)
    plt.xlabel(f"log10({column}/s)")
    plt.yscale('log')
    plt.legend(title='alloc_cpus', fontsize='x-small')

    plt.subplot(ny, nx, 5)
    column = "run_cpu_frac"
    for count in cpu_counts:
        df = df0.query(f"cpu_count == {count}")
        plt.hist(df[column], bins=100, label=str(int(count)), alpha=0.5)
    plt.xlabel(f"run_cpu_time / run_wall_time")
    plt.yscale('log')
    plt.legend(title='alloc_cpus', fontsize='x-small')

    plt.subplot(ny, nx, 6)
    for ccd_type in ccd_types:
        df = df0.query(f"ccd_type == '{ccd_type}'")
        plt.hist(df['max_rss'], bins=100, label=ccd_type, alpha=0.5)
    plt.legend(fontsize='x-small')
    plt.xlabel('max_rss (GB)')
    plt.yscale('log')

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.suptitle(title)
