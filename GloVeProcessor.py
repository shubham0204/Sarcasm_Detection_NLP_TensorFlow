
import numpy as np
import pickle

vocab_size = 18794
glove_path = "C:\\Users\Equip\Desktop\Shubham's Stuff\glove.6B\glove.6B.50d.txt"
tokenizer_path = "glove_embedding/tokenizer.pkl"
output_file_path = 'glove_embedding/embedding.npy'
output_dim = 50

embeddings_index = dict()
f = open( glove_path , encoding='utf8' )
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

tokenizer = pickle.load( open( tokenizer_path , 'rb' ) )

embedding_matrix = np.zeros((vocab_size, output_dim))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

np.save( output_file_path , embedding_matrix )


