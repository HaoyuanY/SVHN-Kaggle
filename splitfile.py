import pandas as pd
import scipy.io as sio

filename = 'dataset/extra.csv'

chunksize = 14*10**5
temp = 1

for chunk in pd.read_csv(filename, chunksize=chunksize):
	b=chunk.as_matrix(columns=None)
	y=b[:,0]
	x=b[:,1:]
	sio.savemat('train'+str(temp)+'.mat')
	temp = temp + 1