import numpy as np
import pandas as pd


csv_data = pd.read_csv("../data/double11_1020_1120.csv")
csv_data *= 100.0
# csv_data.fillna(0.0,inplace=True)
# print(csv_data.isnull().any())
csv_data_u = csv_data.round(5).drop_duplicates(subset=csv_data.columns[1:],keep='first')
# csv_data_u = csv_data

csv_data_u = csv_data_u.sample(n=65536, frac=None, replace=False, weights=None, random_state=None, axis=0)
csv_data_u_cut = csv_data_u.iloc[:,1:]
csv_data_u_float = csv_data_u_cut.astype('float32')
# print(csv_data_u_float.isnull().any())
# print(csv_data_u.iloc[:10,:])
print(csv_data_u_float.shape)

for x in csv_data_u_float.duplicated():
    if (x is True):
        print("duplication exist")
        break

with open("../data/eco_nodes",'wb') as bin_output:
    csv_data_u_float.values.tofile(bin_output)

with open("../data/eco_nodes.csv",'w') as csv_output:
    csv_data_u.to_csv(csv_output)