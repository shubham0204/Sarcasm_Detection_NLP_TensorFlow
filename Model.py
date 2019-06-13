
import time
import tensorflow as tf
import numpy as np

class Classifier( object ):

	def __init__(self , input_length ):
		tf.logging.set_verbosity( tf.logging.ERROR )
		embedding_matrix = np.load( 'glove_embedding/embedding.npy' )
		model_layers = [
			tf.keras.layers.Embedding( embedding_matrix.shape[0] , output_dim=50 , input_length=input_length, trainable=True ,
									weights=[embedding_matrix] ) ,
			tf.keras.layers.Conv1D( 32 , kernel_size=5 , activation="relu",strides=1 , input_shape=( input_length , 50 )),
			tf.keras.layers.Conv1D( 64, kernel_size=5, activation="relu", strides=1),
			tf.keras.layers.MaxPool1D( pool_size=4 , strides=1 ),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense( 256 , activation="relu" ),
			tf.keras.layers.Dropout( 0.5 ) ,
			tf.keras.layers.Dense(2, activation="softmax" )
		]
		self.__model = tf.keras.Sequential( model_layers )
		self.__model.compile( loss=tf.keras.losses.categorical_crossentropy ,
							  optimizer=tf.keras.optimizers.Adam( lr=0.0001 ) ,
							  metrics=[ 'accuracy' ]
							  )


	def fit(self, X, Y, hyperparameters):
		initial_time = time.time()
		self.__model.fit(X, Y,
						 batch_size=hyperparameters['batch_size'],
						 epochs=hyperparameters['epochs'],
						 callbacks=hyperparameters['callbacks'],
						 validation_data=hyperparameters['val_data']
						 )
		final_time = time.time()
		eta = (final_time - initial_time)
		time_unit = 'seconds'
		if eta >= 60:
			eta = eta / 60
			time_unit = 'minutes'
		self.__model.summary()
		print('Elapsed time acquired for {} epoch(s) -> {} {}'.format(hyperparameters['epochs'], eta, time_unit))


	def evaluate(self, test_X, test_Y):
		return self.__model.evaluate(test_X, test_Y)


	def predict(self, X):
		predictions = self.__model.predict(X)
		return predictions


	def save_model(self, file_path):
		self.__model.save(file_path)


	def load_model(self, file_path):
		self.__model = tf.keras.models.load_model(file_path)

