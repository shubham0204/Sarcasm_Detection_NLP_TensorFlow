
import tensorflow as tf
import json
import pickle

model = tf.keras.models.load_model( 'models/model.h5' )
embedding_matrix = model.layers[0].get_weights()[0]
print( 'Embedding Shape ~> {}'.format( embedding_matrix.shape ) )

word_index : dict = pickle.load( open( 'glove_embedding/tokenizer.pkl' , 'rb' ) ).word_index
word_index_2 = dict()
for word , index in word_index.items():
	word_index_2[ index ] = word
word_index = word_index_2
embedding_dict = dict()

for i in range( len( embedding_matrix ) - 1 ):
	embedding_dict[ word_index[ i + 1 ] ] = embedding_matrix[ i + 1 ].tolist()

with open( 'android/embedding.json' , 'w' ) as file:
	json.dump( embedding_dict , file )

new_model = tf.keras.Sequential( model.layers[ 1 : ] )
new_model.save( 'android/no_embedding_model.h5' )

converter = tf.lite.TFLiteConverter.from_keras_model_file( 'android/no_embedding_model.h5')
converter.post_training_quantize = True
tflite_buffer = converter.convert()
open( 'android/model.tflite' , 'wb' ).write( tflite_buffer )