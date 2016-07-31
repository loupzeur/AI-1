from __future__ import print_function
import re
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
import pickle
import os.path
import keras

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.visualize_util import plot
from keras.utils.layer_utils import layer_from_config
from keras.models import model_from_json

vocabulary_size = 20000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
trainingSet = "tst-clean.data"

nb_data = 4382
nb_length = 150
nb_classi = 3

def TrainingData():
	if os.path.isfile(trainingSet): 
		data = pickle.load( open( trainingSet, "rb" ) )
		return data['X'],data['Y'],data['iw'],data['wi']
	return TrainingDataFromTXT()

def TrainingDataFromTXT():
	print("Reading Log file...")
	y_train = []
	ret = []
	with open('tst2.data', 'r') as f:
		for line in f:
			data = re.sub(r'[\W.:]+', ' ', line)
			sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in data.split(' ')])
			sentences = ["%s" % (x) for x in sentences]
			ret.append(sentences)
			y_train.append([TestXSS(line),TestSQL(line),TestOther(line)])
	print ("Parsed %d sentences." % (len(ret)))

	# Tokenize the sentences into words
	tokenized_sentences = ret #[nltk.word_tokenize(sent) for sent in ret]
	 
	# Count the word frequenciesx
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	print ("Found %d unique words tokens." % len(word_freq.items()))
	 
	# Get the most common words and build index_to_word and word_to_index vectors
	vocab = word_freq.most_common(vocabulary_size-1)
	index_to_word = [x[0] for x in vocab]
	#index_to_word.append(unknown_token)
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

	print ("Using vocabulary size %d." % vocabulary_size)
	print ("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
	 
	# Replace all words not in our vocabulary with the unknown token
	for i, sent in enumerate(tokenized_sentences):
	    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
	 
	print ("\nExample sentence: '%s'" % ret[0])
	print ("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
	 
	# Create the training data
	X_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
	#y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

	pickle.dump({"X":X_train,"Y":y_train,"iw":index_to_word,"wi":word_to_index},open(trainingSet,"wb"))
	return X_train,y_train,index_to_word,word_to_index

def TestXSS(str):
	reg = [
		": XSS Attack Detected",
		": IE XSS Filters - Attack Detected.",
		": Last Matched Message: XSS Attack Detected",
		": XSS Filter - Category 3: Javascript URI Vector",
		": Last Matched Message: IE XSS Filters - Attack Detected."
	]
	for r in reg:
		if str.find(r)>=0:
			return 1
	return 0

def TestSQL(str):
	reg = [
		": SQL Comment Sequence Detected.",
		": SQL Injection Attack",
		": Last Matched Message: SQL Injection Attack",
		": Last Matched Message: Restricted SQL Character Anomaly Detection Alert - Total # of special characters exceeded",
		": SQL Injection Attack: Common Injection Testing Detected",
		": SQL Injection Attack: SQL Operator Detected",
		": 981243-Detects classic SQL injection probings 2/2",
		": 981245-Detects basic SQL authentication bypass attempts 2/3",
		": Last Matched Message: 981257-Detects MySQL comment-/space-obfuscated injections and backtick termination",
		": 981257-Detects MySQL comment-/space-obfuscated injections and backtick termination",
		": Last Matched Message: 981245-Detects basic SQL authentication bypass attempts 2/3",
		": Last Matched Message: 981243-Detects classic SQL injection probings 2/2",
		": Restricted SQL Character Anomaly Detection Alert - Total # of special characters exceeded"
	]
	for r in reg:
		if str.find(r)>=0:
			return 1
	return 0

def TestOther(str):
	reg = [
		": Empty User Agent Header",
		": Meta-Character Anomaly Detection Alert - Repetative Non-Word Characters",
		": Request Indicates a Security Scanner Scanned the Site",
		": Request Missing a Host Header",
		": Pragma Header requires Cache-Control Header for HTTP/1.1 requests.",
		": Host header is a numeric IP address",
		": Rogue web site crawler",
		": Request Missing a User Agent Header",
		": Range: field exists and begins with 0.",
		": Request Missing an Accept Header",
		": HTTP protocol version is not allowed by policy"
	]
	for r in reg:
		if str.find(r)>=0:
			return 1
	return 0

X,Y,iw,wi=TrainingData()
#print(X)
#print(Y)
#print(iw)
#print(wi)

############################## AI STUFF
'''Train a recurrent convolutional network on the IMDB sentiment
classification task.
GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python data-cleanup.py
Get to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''


# Embedding
max_features = 10000
maxlen = nb_length
embedding_size = nb_length

# Convolution
filter_length = 4
nb_filter = 64
pool_length = 3

# LSTM
lstm_output_size = 70

# Training
batch_size = 5
nb_epoch = 2
# 3 éléments de classification : XSS, SQL, Fpt
nb_classes = 3

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

#print('Loading data...')
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)

def SaveModel(model):
	#pickle.dump({"config":model.get_config(),"weight":model.get_weights()},open(trainingSet+".model","wb"))
	print('Sauvegarde du modele')
	model_json = model.to_json()
	with open(trainingSet+".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(trainingSet+".h5")

def TrainModel(X,Y):
	#if os.path.isfile(trainingSet+'.model'): 
	#	data = pickle.load( open( trainingSet+'.model', "rb" ) )
	#	print(data)
	#	model = layer_from_config(data['config'])
	#	model.set_weights(data['weight'])
	#	return model
	if os.path.isfile(trainingSet+'.json'):
		print('Récupération du modele...') 
		json_file = open(trainingSet+".json", 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights(trainingSet+".h5")
		return loaded_model
	return TrainModel_Data(X,Y)

def TrainModel_Data(X,Y):
	X_train = sequence.pad_sequences(np.array(X), maxlen=maxlen)
	X_test = sequence.pad_sequences(np.array(X[:100]), maxlen=maxlen)

	y_train = np.array(Y)
	y_test  = np.array(Y[:100])

	print(len(X_train), 'train sequences')
	print(len(X_test), 'test sequences')

#	print(X_train)
#	print(y_train)

	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)

	print('y_train shape:', y_train.shape)
	print('y_test shape:', y_test.shape)

	print('Build model...')

	model = Sequential()
	model.add(Embedding(max_features, embedding_size, input_length=maxlen))
	model.add(Dropout(0.25))
	model.add(Convolution1D(nb_filter=nb_filter,
	                        filter_length=filter_length,
	                        border_mode='valid',
	                        activation='relu',
	                        subsample_length=nb_classes))
	model.add(MaxPooling1D(pool_length=pool_length))
	model.add(LSTM(lstm_output_size))
	model.add(Dense(nb_classes))
	model.add(Activation('sigmoid'))
	#model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam')
	print('Train...')
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(X_test, y_test))
	score = model.evaluate(X_test, y_test)
	print('Test score:', score)
	SaveModel(model)
	return model

print(X[:1])
#print(X[0])
t=[iw[t] for t in X[0]]
#print(t)
model = TrainModel(X,Y)
tst = model.predict(sequence.pad_sequences(np.array(X[:1]), maxlen=maxlen))
print(tst)
#weight=model.get_weights()[4][0]
#itera=0
#diff=len(weight)-len(X[0])
#print("Weight : %d X[0] : %d" %(len(weight),len(X[0])))

#for w in weight[diff:]:
#	print("Score : %f IndexW : %d Word : %s = %f" %(w,X[0][itera],iw[X[0][itera]],w*X[0][itera]))
#	itera=itera+1

#print(model.get_config())
#print(model.get_weights())