import sys
from collections import defaultdict
from itertools import pairwise
import multiprocessing
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import lsst.daf.butler as daf_butler
import lsst.geom
import lsst.sphgeom
from lsst.summit.utils import ConsDbClient


def compute_visit_overlaps(df0):
    data = defaultdict(list)
    for _, row in df0.iterrows():
        region_str = row['s_region']
        region = lsst.sphgeom.Region.from_ivoa_pos(
            "".join(region_str.split("ICRS")).upper())
        coord_list = [lsst.geom.SpherePoint(_) for _ in region.getVertices()]
        tract_patch_list = skymap.findTractPatchList(coord_list)
        for tract, patch_list in tract_patch_list:
            for patch in patch_list:
                data['band'].append(row.band)
                data['visit'].append(row.visit_id)
                data['detector'].append(row.detector)
                data['tract'].append(tract.tract_id)
                data['patch'].append(patch.sequential_index)
    return pd.DataFrame(data)


# Get the skymap
repo = "/repo/main"
butler = daf_butler.Butler(repo)
skymap = butler.get("skyMap", name="lsst_cells_v1", collections=["skymaps"])

with open("/sdf/home/j/jchiang/.consdb_token") as fobj:
    token = fobj.read()
client = ConsDbClient(f"https://user:{token}@usdf-rsp.slac.stanford.edu/consdb")
instrument = "lsstcam"
schema_name = f"cdb_{instrument}"

columns = ("v.band, v.visit_id, v.exp_midpt_mjd, v.exp_time, v.sky_rotation, "
           "cv.detector, cv.s_region")

constraints = "v.day_obs>=20250501 and v.day_obs<=20250609 and v.img_type='science' and (v.target_name in ('M49', 'Rubin_SV_225_-40', 'COSMOS', 'New_Horizons')) and (cv.detector < 189 and (cv.detector not in (0, 20, 27, 65, 123, 161, 168, 188, 78, 79, 80, 120, 121, 122, 158, 187))) and (v.visit_id not in (2025041700761,2025041800615,2025042000501,2025042300237,2025042700338,2025042700381,2025042800106,2025042800234,2025042900186,2025042900190,2025042900191,2025042900192,2025042900193,2025042900210,2025042900211,2025042900229,2025042900286,2025042900484,2025050100794,2025050200275,2025050200579,2025050200699,2025050400309,2025050400329,2025050400538,2025050400595,2025050600759,2025050600760,2025050600761,2025050600781,2025051500148,2025052000146,2025052000152,2025052000321,2025052000357,2025052000358,2025052000372,2025052100118,2025052100130,2025052501031,2025053100372,2025060100555,2025060300096,2025060300097,2025060300098,2025060300099,2025060300100,2025060300102,2025060300103,2025060300106,2025060300107,2025060300108,2025060300109,2025060300110,2025060300111,2025060300112,2025060300113,2025060300114,2025060300115,2025060300116,2025060300117,2025060300118))"

query = (f"select {columns} from cdb_LSSTCam.ccdvisit1 as cv, "
         f"cdb_LSSTCam.visit1 as v where cv.visit_id=v.visit_id "
         f"and {constraints}")

df0 = client.query(query).to_pandas()
print(len(df0))

processes = 32
indexes = np.linspace(0, len(df0), processes+1, dtype=int)
with multiprocessing.Pool(processes=processes) as pool:
    workers = []
    for imin, imax in pairwise(indexes):
        workers.append(pool.apply_async(compute_visit_overlaps,
                                        (df0[imin:imax],)))
    pool.close()
    pool.join()
    df1 = pd.concat([worker.get() for worker in workers])
df1.to_parquet("DM-51933_visit_skymap_overlaps.parquet")
