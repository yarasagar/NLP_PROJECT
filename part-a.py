import os
from collections import Counter
import math
import pickle

def removeUnWanted(name):
	newWord=[]
	stop = open('stopwords.txt','r')
	for word in stop:
		newWord.append(word[ :-1])	#remove \n in list
	inFile = open(name,'r')
	data = inFile.read()
	inFile.close()
	for stopWord in newWord:
		data = data.replace(" "+stopWord+" "," ")
		data = data.replace("\n"+stopWord+" ","\n")
		data = data.replace(" "+stopWord+"\n","\n")
	data = data.lower()
	data = data.replace("\"", "")
	data = data.replace(",", "")
	data = data.replace(".", " ")
	data = data.replace("!", " ")
	data = data.replace("/", " ")
	data = data.replace("-", " ")
	data = data.replace("?", " ")
	data = data.replace("`", " ")
	data = data.replace("\'", " ")
	data = data.replace(";", " ")
	data = data.replace("&", " ")
	data = data.replace("\\", " ")
	data = data.replace("*", " ")
	data = data.replace("$", "")
	data = data.replace(":", " ")
	data = data.replace(" s ", " ")
	data = data.replace("\ns ", "")
	data = data.replace(" d "," ")

	s='0123456789'
	for i in s:
		data = data.replace(i,"")	
	outFile = open('out_'+name,'w')
	outFile.write(data)
	outFile.close()

def destroy(fname):
	if os.path.exists(fname):
		os.remove(fname)

def split10(part):
	infile = open(part,'r')
	for i in range(1,11):
		destroy(str(i)+part)
		outfile = open(str(i)+part,'w')
		for j in range(50):
			outfile.write(infile.readline())
		outfile.close()
	infile.close()

def pro2(doc,clas):
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	res = 0
	doc = doc.split()
	theta = tot_freq_neg
	if clas == 1:
		theta = tot_freq_pos
	for word in doc:
		if word in voc:
			res += db[word][clas]
		else:
			res += math.log(1)-math.log(theta+voc_size)
	return res

def predict(doc):
	if pro2(doc,1)>=pro2(doc,0):
		return 1 #pos
	else:
		return 0 #neg

def pro(word,flag):
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	db1 = neg
	tot = tot_freq_neg
	if flag==1:
		db1 = pos
		tot = tot_freq_pos
	if word in db1:
		return math.log(db1[word]+1)-math.log(tot+voc_size)
	else: 
		return math.log(1)-math.log(tot+voc_size)

def  create_model():
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	db = {}
	for word in voc:
		db[word]=[pro(word,0),pro(word,1)]
	pickleOut(db,'MNBmodel')
	
def total(db1):
	c=0
	for word in db1:
		c += db1[word]
	return c

def pickleOut(temp,fname):
	pickle_out = open(fname+'.p', 'wb')
	pickle.dump(temp, pickle_out)
	pickle_out.close()

def pickleIn(fname):
	pickle_in = open(fname+'.p','rb')
	db1 = pickle.load(pickle_in)
	pickle_in.close()
	return db1

def lis2dic(l):
	counts = Counter(l)
	counts = {i:counts[i] for i in counts if counts[i]>=2}
	return counts

def tokenize(fname):
	infile = open(fname,'r')
	l=[]
	for line in infile.readlines():
		ls = line.split()
		ls = [i.strip('\n') for i in ls if i!='']
		l += ls
	return l

def verify():
	global pos,neg,voc,db,tot_freq_pos,tot_freq_neg,voc_size
	sum_acc =0
	for i in range(1,11):
		pos={}
		neg={}
		voc={}
		db={}
		tot_pos_freq=0
		tot_neg_freq=0
		voc_size=0
		traindat1,traindat2 = '',''
		for j in range(1,11):
			if j != i:
				testpos = open(str(j)+'pos2.txt')	
				traindat1 += testpos.read()
				testpos.close()
				testneg = open(str(j)+'neg2.txt')
				traindat2 += testneg.read()
				testneg.close()
		destroy('train_pos2.txt')
		intrain = open('train_pos2.txt','w')
		intrain.write(traindat1)
		intrain.close()
		destroy('train_neg2.txt')
		intrain = open('train_neg2.txt','w')
		intrain.write(traindat2)
		intrain.close()

		ls =  (traindat1 + traindat2).split('\n')
		print 'length of training data :',len(ls)
		pfl =tokenize('train_pos2.txt')
		pos = lis2dic(pfl)
		nfl = tokenize('train_neg2.txt')
		neg = lis2dic(nfl)
		voc = lis2dic(pfl+nfl)
		voc_size = len(voc)
		tot_freq_pos = total(pos)
		tot_freq_neg = total(neg)
		#print voc_size,tot_freq_pos,tot_freq_neg
		create_model()
		db = pickleIn('MNBmodel')
		fp,fn,tp,tn = 0,0,0,0
		tes = open(str(i)+'pos2.txt','r').read().split('\n')
		for doc in tes:
			if predict(doc)==1:
				tp += 1
			else:
				fn += 1
		testl = open(str(i)+'neg2.txt','r').readlines()
		for doc in testl:
			if predict(doc)==0:
				tn += 1
			else:
				fp += 1
		accuracy = (tp+tn)/float(tp+tn+fn+fp)
		print 'accuracy',accuracy
		sum_acc  +=  accuracy
	avg_acc = sum_acc/10.0
	print avg_acc

removeUnWanted('pos2.txt')
removeUnWanted('neg2.txt')
split10('out_pos2.txt')
split10('out_neg2.txt')
verify()