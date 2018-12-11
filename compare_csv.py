import pandas as pd
import numpy as np
import os

csv1 = 'weighted_ensemble_fantastic4_allout.csv'
csv2 = 'weighted_ensemble_fantastic4.csv'

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

df = pd.concat((df1, df2), axis=1)

df['compare'] = (df1['category'] != df2['category'])

diffs = np.array(df[df['compare'] == True].index.tolist()) + 1

print(diffs)
