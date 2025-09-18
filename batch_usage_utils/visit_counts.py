import os
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import lsst.geom
from lsst.pipe.base import QuantumGraph
import lsst.sphgeom
from lsst.summit.utils import ConsDbClient


__all__ = ["query_consdb", "compute_visit_overlaps",
           "extract_visits_per_patch", "VisitCounts"]


_TOKEN_FILE = "/sdf/home/j/jchiang/.consdb_token"


def query_consdb(constraints, token_file=_TOKEN_FILE, instrument="lsstcam"):
    with open(token_file) as fobj:
        token = fobj.read()
    client = ConsDbClient(
        f"https://user:{token}@usdf-rsp.slac.stanford.edu/consdb")
    schema_name = f"cdb_{instrument}"
    columns = ("v.band, v.visit_id, v.exp_midpt_mjd, v.exp_time, "
               "v.sky_rotation, cv.detector, cv.s_region")
    query = (f"select {columns} from {schema_name}.ccdvisit1 as cv, "
             f"{schema_name}.visit1 as v where cv.visit_id=v.visit_id "
             f"and {constraints}")
    return client.query(query).to_pandas()


def compute_visit_overlaps(consdb_df, skymap):
    data = defaultdict(list)
    for _, row in consdb_df.iterrows():
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


def extract_visits_per_patch(qgraph_files, coadd_task="assembleDeepCoadd",
                             warp_dstype="direct_warp"):
    """
    Extract the number of visits per patch from QuantumGraph files
    based in the warp inputs to the coadd task instances.
    """
    data = defaultdict(list)
    for qgraph_file in qgraph_files:
        print(os.path.basename(qgraph_file), flush=True)
        qg = QuantumGraph.loadUri(qgraph_file)
        for node in tqdm(qg.graph.nodes):
            if node.taskDef.label != coadd_task:
                continue
            dataId = node.quantum.dataId  # noqa: N806
            dimensions = dataId.dimensions.to_simple()
            for dimension in dimensions:
                data[dimension].append(dataId[dimension])
            data['num_visits'].append(len(node.quantum.inputs[warp_dstype]))
    return pd.DataFrame(data)


class VisitCounts:
    """
    Class to keep track of visits per patch-tract-band combinations
    for a given set of visits and skymap.
    """
    def __init__(self, overlaps_file):
        df = pd.read_parquet(overlaps_file)
        self.df = df["visit patch tract band".split()].drop_duplicates()
        self._setup_num_visits_dict()

    def _setup_num_visits_dict(self):
        df0 = pd.DataFrame(self.df)
        df0["num_visits"] = 1
        # Fill dict with num_visits for each patch-tract-band combination.
        index_cols = "patch tract band".split()
        df1 = df0.groupby(index_cols)["num_visits"].sum().reset_index()
        self._num_visits = dict(zip(zip(*[df1[_] for _ in index_cols]),
                                    df1["num_visits"]))
        # Add entries for num_visits for each patch-tract combination.
        df2 = df1.groupby(["patch", "tract"])["num_visits"].sum().reset_index()
        self._num_visits.update(
            dict(zip(zip(df2["patch"], df2["tract"], [None]*len(df2)),
                     df2["num_visits"]))
        )

    def __call__(self, patch, tract, band=None):
        return self._num_visits[(patch, tract, band)]
