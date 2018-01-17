import math
import json
from pandas import DataFrame, Series
import pandas as pd;
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = 'Input/datasets/bitly_usagov/example.txt'
    records = [json.loads(line) for line in open(path)]
    time_zones = [rec['tz'] for rec in records if 'tz' in rec]
    frame = DataFrame(records)
    tz_counts = frame['tz'].value_counts()
    clean_tz = frame['tz'].fillna('Missing')
    clean_tz[clean_tz == ''] = 'Unknown'
    tz_counts = clean_tz.value_counts()
    results = Series([x.split()[0] for x in frame.a.dropna()])
    cframe = frame[frame.notnull()]
    operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
    by_tz_os = cframe.groupby(['tz', operating_system])
    agg_counts = by_tz_os.size().unstack().fillna(0)
    indexer = agg_counts.sum(1).argsort()
    count_subset = agg_counts.take(indexer)[-10:]
    count_subset.plot(kind='barh', stacked=True)
    plt.show()