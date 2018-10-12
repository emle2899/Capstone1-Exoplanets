import pandas as pd
import numpy as np
from tabulate import tabulate

df_kep = pd.read_csv('data/exoplanet.csv')

# drop all data containing error, naming, id numbers and KIC Parameters
df_kep = df_kep.loc[:,~df_kep.columns.str.endswith('_err1')]
df_kep = df_kep.loc[:,~df_kep.columns.str.endswith('_err2')]
df_kep = df_kep.loc[:,~df_kep.columns.str.endswith('_name')]
df_kep = df_kep.loc[:,~df_kep.columns.str.endswith('mag')]
df_kep = df_kep.loc[:,~df_kep.columns.str.endswith('num')]
df_kep = df_kep.loc[:,~df_kep.columns.str.contains('fpflag')]
df_kep = df_kep.loc[:,~df_kep.columns.str.contains('id')]
df_kep = df_kep.loc[:,~df_kep.columns.str.contains('time')]
df_kep = df_kep.loc[:,~df_kep.columns.str.contains('model')]

# drop uncorrelated columns such as positional, confidence columns, and eccentricity
df_kep.drop(['koi_period','ra','dec','koi_score','koi_eccen','koi_tce_delivname','koi_count'], axis=1,inplace = True)

# drop columns with only NaN values in it
df_kep = df_kep.dropna(axis=1, how='all')
# replace remaining NaN values with nonNaN mean
df_kep = df_kep.fillna(df_kep.mean(skipna=True))

# change pdisposition to binary
df_kep = df_kep.replace({'koi_pdisposition' : {'FALSE POSITIVE':0, 'CANDIDATE':1}})
# Running training on pdisposition only
df_kep2 = df_kep.drop(['koi_disposition'], axis=1)

# read into new cleaned csv file
df_kep2.to_csv('/Users/emily/Documents/galvanize/code_sprints/capstone/data/cleaned.csv')
