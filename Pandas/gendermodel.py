import pandas as pd
import numpy as np
import pylab as P

df = pd.read_csv('../Data/train.csv', header=0)
print 'Drawing histogram...'
df.Age.dropna().hist(bins=16, range=(0,80), alpha=0.5)
P.show()
print 'Done...'
