repo = "embargo_new"
butler = daf_butler.Butler(repo)
collections = sorted(butler.registry.queryCollections(
    "u/lsstccs/ptc_*",
    collectionTypes=[daf_butler.CollectionType.CHAINED]))
print(len(collections), flush=True)

collections = ["u/jchiang/ptc_E1880_w_2024_48"]

dstype = 'cpPtcIsr_metadata'

for collection in collections[:1]:
    acq_run = os.path.basename(collection).split('_')[1]
    outfile = f"{dstype}_{acq_run}_md_stats_isrTaskLSST.parquet"
    if os.path.isfile(outfile):
        continue
    refs = list(butler.registry.queryDatasets(dstype, collections=collection))
    print(collection, len(refs), flush=True)
    data = defaultdict(list)
    i = 0
    for ref in refs:
        i += 1
        if i % (len(refs)//20) == 0:
            print('.', end="", flush=True)
        md = butler.get(ref)
        try:
            data = extract_metadata(md, data)
        except:
            continue
        data['acq_run'].append(acq_run)
        for key in ("detector", "exposure"):
            data[key].append(ref.dataId[key])
    print("!", flush=True)
    df0 = pd.DataFrame(data)
    df0.to_parquet(outfile)
