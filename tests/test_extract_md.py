from collections import defaultdict
import unittest
import pandas as pd
from batch_usage_utils import extract_metadata


class ExtractMetadataTestCase(unittest.TestCase):
    """TestCase class for extracting task metadata"""

    def setUp(self):
        self.md = {"quantum": {"prepUtc": "2025-10-24T23:49:52.923238+00:00",
                               "initUtc": "2025-10-24T23:49:52.957417+00:00",
                               "startUtc": "2025-10-24T23:49:52.959856+00:00",
                               "endUtc": "2025-10-24T23:50:31.154800+00:00",
                               "endMaxResidentSetSize": 4889600000,
                               "endMajorPageFaults": 1,
                               "prepCpuTime": 28.980089625,
                               "initCpuTime": 28.994917075,
                               "startCpuTime": 28.997206767,
                               "endCpuTime": 66.57026813,
                               "nodeName": "sdfmilan247",
                               "butler_metrics": {'time_in_put': 3.02,
                                                  'time_in_get': 6.61,
                                                  'n_get': 11,
                                                  'n_put': 2}}}

    def tearDown(self):
        pass

    def test_extract_metadata(self):
        data = defaultdict(list)
        df = pd.DataFrame(extract_metadata(self.md, data))
        row = df.iloc[0]
        self.assertAlmostEqual(row.run_wall_time, 38.231562)
        self.assertAlmostEqual(row.start_utc, 60972.99297364)
        self.assertAlmostEqual(row.end_utc, 60972.99341614)
        self.assertAlmostEqual(row.max_rss, 4.55379486)
        self.assertEqual(row.major_page_faults, 1)
        self.assertAlmostEqual(row.prep_cpu_time, 0.01482745)
        self.assertAlmostEqual(row.init_cpu_time, 0.002289692)
        self.assertAlmostEqual(row.run_cpu_time, 37.573061363)
        self.assertEqual(row.node_name, "sdfmilan247")
        self.assertEqual(row.time_in_put, 3.02)
        self.assertEqual(row.time_in_get, 6.61)
        self.assertEqual(row.n_get, 11)
        self.assertEqual(row.n_put, 2)


if __name__ == '__main__':
    unittest.main()
