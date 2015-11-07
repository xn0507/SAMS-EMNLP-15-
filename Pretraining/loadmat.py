from scipy.io import loadmat
import numpy 
import cPickle as pickle
data = loadmat('test.mat')['test']
allSNum, allSStr, allSTree, allSKids = data[0][0]
allSNum = allSNum.flatten()
allSKids = allSKids.flatten()

SNum = []
SKids = []
for i in range(len(allSNum)):
	sentence = allSNum[i].flatten().tolist()
	SNum.append(sentence)
#matlab index starts from 1, we need to convert it to fit python
for i in range(len(allSKids)):
	kids = allSKids[i].astype(int)-1
	kids = kids.tolist()
	SKids.append(kids)

t = (SNum, SKids)
pfile = open('input.pkl','wb')
pickle.dump(t, pfile)

pfile.close()




