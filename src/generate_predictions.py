from sklearn.neighbors import NearestNeighbors as knnsearch
import scipy.io
import numpy as np
import json

outer_dict = json.load(open('data/temporal_test.json','rb'))
data_w2v = scipy.io.loadmat('data/all_w2v.mat')['data_w2v']

nouns = []
with open('data/all_noun.txt','rb') as fp:
	for line in fp.readlines():
		nouns.append(line.strip('\n')) 

pred = np.load('out/pred_release.npz')['arr_0']
knn = knnsearch(n_neighbors=1, algorithm='brute', metric='cosine', n_jobs=3)
knn.fit(data_w2v)
for sketch_idx in range(len(pred)):
	sketch_data = outer_dict[str(sketch_idx).zfill(6)]
	file_sequence = sketch_data['file_sequence']
	category = sketch_data['sketch_category']
	no_of_strokes = sketch_data['no_of_strokes']
	human_guess = sketch_data['sequence']
	print("Category: %s" %(category))
	print("No of strokes in sketch: %d" %(no_of_strokes))
	_, knn_idx = knn.kneighbors(X=np.array(pred[sketch_idx]), n_neighbors=1)
	for counter,idx in enumerate(knn_idx):
		print("Stroke no: %d" %(counter))
		print("Machine generated guess: %s" %(nouns[int(idx)]))
		print("Original Human guess: %s" %(human_guess[counter][1]))
	
	print 100*"--"
