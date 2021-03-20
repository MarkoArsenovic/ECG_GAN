from read_data import read_patient
import numpy as np

train_set=["101","106","108","109","112","114","115","116","118","119","122","124","201","203","205","207","208","209","215","220","223","230"]
#train_set=["101"]
test_set=["100","103","105","111","113","117","121","123","200","202","210","212","213","214","219","221","222","228","231","232","233","234"]
#test_set=[]

def build_dataset(mit_bih_label):
	heartbeats=[]
	for patient in train_set:
		heartbeats+=[heartbeat['heartbeat'] for heartbeat in read_patient(patient) if heartbeat['mit_bih']==mit_bih_label]
	return heartbeats

def make_one_hot(label,labels):
	return [1.0 if label==x else 0.0 for x in labels]

def random_shuffle_2(array1,array2):
	for i in range(len(array1)):
		j=np.random.randint(i+1)
		if i!=j:
			array1[i],array1[j]=array1[j],array1[i]
			array2[i],array2[j]=array2[j],array2[i]
	
def random_shuffle_1(array):
	for i in range(len(array)):
		j=np.random.randint(i+1)
		if i!=j:
			array[i],array[j]=array[j],array[i]
	
def build_validation(labels):
	heartbeats,one_hot=[],[]
	for patient in train_set:
		for heartbeat in read_patient(patient):
			if heartbeat['mit_bih'] in labels:
				heartbeats.append(heartbeat['heartbeat'])
				one_hot.append(make_one_hot(heartbeat['mit_bih'],labels))
	random_shuffle_2(heartbeats,one_hot)
	return heartbeats,one_hot

def build_testset(labels):
	heartbeats,one_hot=[],[]
	for patient in test_set:
		for heartbeat in read_patient(patient):
			if heartbeat['mit_bih'] in labels:
				heartbeats.append(heartbeat['heartbeat'])
				one_hot.append(make_one_hot(heartbeat['mit_bih'],labels))
	random_shuffle_2(heartbeats,one_hot)
	return heartbeats,one_hot

def build_mixed_dataset(labels):
	heartbeats={}
	for label in labels:
		heartbeats[label]=[]
	for patient in train_set+test_set:
		for heartbeat in read_patient(patient):
			if heartbeat['mit_bih'] in labels:
				heartbeats[heartbeat['mit_bih']].append(heartbeat['heartbeat'])
	train_x,train_y=[],[]
	validation_x,validation_y=[],[]
	test_x,test_y=[],[]
	for i,label in enumerate(labels):
		random_shuffle_1(heartbeats[label])
		size=len(heartbeats[label])
		if size==0:
			continue
		test=min((size+9)//10,size-1)
		validation=min((size+4)//5,size-test-1)
		train=size-test-validation
		for i in range(test):
			test_x.append(heartbeats[label][i])
			test_y.append(make_one_hot(label,labels))
		for i in range(test,test+validation):
			validation_x.append(heartbeats[label][i])
			validation_y.append(make_one_hot(label,labels))
		for i in range(test+validation,size):
			train_x.append(heartbeats[label][i])
			train_y.append(make_one_hot(label,labels))
	random_shuffle_2(train_x,train_y)
	random_shuffle_2(validation_x,validation_y)
	random_shuffle_2(test_x,test_y)
	return (train_x,train_y),(validation_x,validation_y),(test_x,test_y)


if __name__=="__main__":
	#dataset=build_dataset("N")
	#print(len(dataset))
	train=[]
	for patient in train_set:
		train+=read_patient(patient)
	test=[]
	for patient in test_set:
		test+=read_patient(patient)
	cnt_train={}
	cnt_test={}
	for heartbeat in train:
		if heartbeat['mit_bih'] in cnt_train:
			cnt_train[heartbeat['mit_bih']]+=1
		else:
			cnt_train[heartbeat['mit_bih']]=1
	for heartbeat in test:
		if heartbeat['mit_bih'] in cnt_test:
			cnt_test[heartbeat['mit_bih']]+=1
		else:
			cnt_test[heartbeat['mit_bih']]=1
	print("Train:")
	print(cnt_train)
	print("Test:")
	print(cnt_test)
