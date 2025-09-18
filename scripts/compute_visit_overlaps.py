from itertools import pairwise
import multiprocessing
import numpy as np
import pandas as pd
import lsst.daf.butler as daf_butler
from batch_usage_utils import query_consdb, compute_visit_overlaps


constraints = "v.day_obs>=20250501 and v.day_obs<=20250609 and v.img_type='science' and (v.target_name in ('M49', 'Rubin_SV_225_-40', 'COSMOS', 'New_Horizons')) and (cv.detector < 189 and (cv.detector not in (0, 20, 27, 65, 123, 161, 168, 188, 78, 79, 80, 120, 121, 122, 158, 187))) and (v.visit_id not in (2025041700761,2025041800615,2025042000501,2025042300237,2025042700338,2025042700381,2025042800106,2025042800234,2025042900186,2025042900190,2025042900191,2025042900192,2025042900193,2025042900210,2025042900211,2025042900229,2025042900286,2025042900484,2025050100794,2025050200275,2025050200579,2025050200699,2025050400309,2025050400329,2025050400538,2025050400595,2025050600759,2025050600760,2025050600761,2025050600781,2025051500148,2025052000146,2025052000152,2025052000321,2025052000357,2025052000358,2025052000372,2025052100118,2025052100130,2025052501031,2025053100372,2025060100555,2025060300096,2025060300097,2025060300098,2025060300099,2025060300100,2025060300102,2025060300103,2025060300106,2025060300107,2025060300108,2025060300109,2025060300110,2025060300111,2025060300112,2025060300113,2025060300114,2025060300115,2025060300116,2025060300117,2025060300118))"
df0 = query_consdb(constraints)

# Get the skymap
repo = "/repo/main"
butler = daf_butler.Butler(repo)
skymap = butler.get("skyMap", name="lsst_cells_v1", collections=["skymaps"])

processes = 32
indexes = np.linspace(0, len(df0), processes+1, dtype=int)
with multiprocessing.Pool(processes=processes) as pool:
    workers = []
    for imin, imax in pairwise(indexes):
        workers.append(pool.apply_async(compute_visit_overlaps,
                                        (df0[imin:imax], skymap)))
    pool.close()
    pool.join()
    df1 = pd.concat([worker.get() for worker in workers])
df1.to_parquet("DM-51933_visit_skymap_overlaps.parquet")
