#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


def transform_data_to_datatime(name):
    dataset = pd.read_excel(name)

    # Convert the Time column to datatime format
    dataset['DATE TIME'] = pd.to_datetime(dataset['DATE TIME'])
    # Deleting rows that have same date time
    same_time = dataset[~dataset['DATE TIME'].dt.round('min').duplicated()]

    same_time.to_excel(name, index=False)


def remove_useless_data(name, new_name):
    dataset = pd.read_csv(name, sep=';', decimal=',')

    # Deleting of values ​​less than 1
    dataset.drop(dataset[(dataset.Fdw < 1) | (dataset.Tdw < 1) | (dataset.Ta < 1) | (dataset.Thwi < 1) | (
                dataset.Pa < 1) | (dataset.RH < 1) | (dataset.Frl < 1) | (dataset.Thwo < 1)].index, inplace=True)
    a = dataset.reset_index(drop=True)
    by_mean = a.groupby(a.index // 5).agg(
        {'Fdw': 'mean', 'Tdw': 'mean', 'Ta': 'mean', 'Thwi': 'mean', 'Pa': 'mean', 'RH': 'mean', 'Frl': 'mean',
         'Thwo': 'mean'})

    # In[ ]:

    by_mean.to_excel(new_name, index=False)
