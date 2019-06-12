
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from Tokenizer import Tokenizer


data = pd.read_json( 'raw_data/data.json' , lines=True )
raw_text , raw_labels =  data[ 'headline' ].values[0:10000] , data[ 'is_sarcastic' ].values[0:10000]

tokenizer = Tokenizer()
tokenizer.fit_on_texts( raw_text )
tokenized_headlines = tokenizer.transform( raw_text )
max_length = max( [ len( x ) for x in  tokenized_headlines ] )
padded_headlines = tf.keras.preprocessing.sequence.pad_sequences( tokenized_headlines , maxlen=max_length , padding='post' )
with open( 'glove_embedding/tokenizer.pkl' , 'wb' ) as file:
	pickle.dump( tokenizer , file )

onehot_labels = tf.keras.utils.to_categorical( raw_labels , num_classes=2 )

train_features , test_features , train_labels, test_labels = train_test_split( np.array(padded_headlines) ,
																			  np.array(onehot_labels) ,
																			  test_size=0.4 )

np.save( 'processed_data/x.npy' , train_features )
np.save( 'processed_data/y.npy' , train_labels )
np.save( 'processed_data/test_x.npy' , test_features )
np.save( 'processed_data/test_y.npy' , test_labels )