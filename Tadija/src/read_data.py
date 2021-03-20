import wfdb
import matplotlib.pyplot as plt

def mit_bih_to_aami(mit_bih):
	if mit_bih in "NLRje":
		return "N"
	elif mit_bih in "aSAJ":
		return "S"
	elif mit_bih in "EV":
		return "V"
	elif mit_bih in "F":
		return "F"
	else:
		return "Q"

def read_patient(index):
	print("Reading heartbeats for patient "+index)
	signals,fields=wfdb.rdsamp(record_name=index,pn_dir="mitdb")
	ann=wfdb.rdann(record_name=index,extension="atr",pn_dir="mitdb",return_label_elements=["symbol","description"],summarize_labels=True)
	mit_bih_label=ann.symbol
	location=ann.sample
	description=ann.description
	heartbeats=[]
	mlii=0
	for i in range(len(fields['sig_name'])):
		if fields['sig_name'][i]=="MLII":
			mlii=i
	L,R=72,144
	for i in range(len(mit_bih_label)):
		#if i<20:
		#	print(location[i],end=' ')
		if location[i]-L<0 or location[i]+R>fields['sig_len']:
			continue
		heartbeats.append({'mit_bih':mit_bih_label[i],'aami':mit_bih_to_aami(mit_bih_label[i]),'heartbeat':signals[location[i]-L:location[i]+R,mlii]})
	#print()
	#print("ann size",len(mit_bih_label))
	return heartbeats

if __name__=="__main__":
	heartbeats=read_patient("101")
	print(len(heartbeats))
	print(len(heartbeats[0]['heartbeat']))
	for num in [0,1]:
		plt.title("MIT-BIH: "+heartbeats[num]['mit_bih']+" AAMI: "+heartbeats[num]['aami'])
		plt.plot(heartbeats[num]['heartbeat'])
		plt.show()