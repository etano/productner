"""Product classifier class"""

import os, json
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Flatten, Dropout, Conv1D, MaxPooling1D, Embedding
from keras.models import load_model, Model
from keras.utils import to_categorical
from sklearn.metrics import classification_report

class ProductClassifier(object):
    """Class which classifies products based on various inputs

       Attributes:
           prefix (str): Model files prefix
           model (keras.model): Keras model
           category_map (dict(str, int)): Map between category names and indices
    """

    def __init__(self, prefix=None):
        """Load in model and category map

        Args:
            prefix (str): Prefix of directory containing model HDF5 file and category map JSON file
        """
        if prefix != None:
            self.load(prefix)
        else:
            self.prefix = 'models/classifier'
            self.model = None
            self.category_map = {}

    def load(self, prefix=None):
        """Load in model and category map

        Args:
            prefix (str): Prefix of directory containing model HDF5 file and category map JSON file
        """
        if prefix != None: self.prefix = prefix
        self.model = load_model(self.prefix+'.h5')
        self.category_map = json.load(open(self.prefix+'.json', 'r'))

    def save(self, prefix=None):
        """Save in model and category map

        Args:
            prefix (str): Prefix of directory containing model HDF5 file and category map JSON file
        """
        if prefix != None: self.prefix = prefix
        self.model.save(self.prefix+'.h5')
        with open(self.prefix+'.json', 'w') as out:
            json.dump(self.category_map, out)

    def index_categories(self, categories):
        """Take a list of possibly duplicate categories and create an index list

        Args:
            categories (list(str)): List of categories
        Returns:
            list(int): List of indices
        """
        print('Indexing categories...')
        indices = []
        for category in categories:
            if not (category in self.category_map):
                self.category_map[category] = len(self.category_map)
            indices.append(self.category_map[category])
        print(('Found %s unique categories.' % len(self.category_map)))
        return indices

    def classify(self, data):
        """Classify by products by text

        Args:
            data (np.array): 2D array representing descriptions of the product and/or product title
        Returns:
            list(dict(str, float)): List of dictionaries of product categories with associated confidence
        """
        prediction = self.model.predict(data)
        all_category_probs = []
        for i in range(prediction.shape[0]):
            category_probs = {}
            for category in self.category_map:
                category_probs[category] = prediction[i,self.category_map[category]]
            all_category_probs.append(category_probs)
        return all_category_probs

    def get_labels(self, categories):
        """Create labels from a list of categories

        Args:
            categories (list(str)): A list of product categories
        Returns:
            (list(int)): List of indices
        """
        indexed_categories = self.index_categories(categories)
        labels = to_categorical(np.asarray(indexed_categories))
        return labels

    def compile(self, tokenizer, glove_dir='./data/', embedding_dim=100, dropout_fraction=0.0, kernal_size=5, n_filters=128):
        """Compile network model for classifier

        Args:
            glove_file (str): Location of GloVe file
            embedding_dim (int): Size of embedding vector
            tokenizer (WordTokenizer): Object used to tokenize orginal texts
            dropout_fraction (float): Fraction of randomly zeroed weights in dropout layer
            kernal_size (int): Size of sliding window for convolution
            n_filters (int): Number of filters to produce from convolution
        """
        # Load embedding layer
        print('Loading GloVe embedding...')
        embeddings_index = {}
        f = open(os.path.join(glove_dir, 'glove.6B.'+str(embedding_dim)+'d.txt'), 'r')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print(('Found %s word vectors.' % len(embeddings_index)))

        # Create embedding layer
        print('Creating embedding layer...')
        embedding_matrix = np.zeros((len(tokenizer.tokenizer.word_index) + 1, embedding_dim))
        for word, i in list(tokenizer.tokenizer.word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        embedding_layer = Embedding(len(tokenizer.tokenizer.word_index) + 1,
                                    embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=tokenizer.max_sequence_length,
                                    trainable=False)

        # Create network
        print('Creating network...')
        sequence_input = Input(shape=(tokenizer.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Dropout(dropout_fraction)(embedded_sequences)
        x = Conv1D(n_filters, kernal_size, activation='relu')(x)
        x = MaxPooling1D(kernal_size)(x)
        x = Conv1D(n_filters, kernal_size, activation='relu')(x)
        x = MaxPooling1D(kernal_size)(x)
        x = Conv1D(n_filters, kernal_size, activation='relu')(x)
        x = MaxPooling1D(int(x.shape[1]))(x)  # global max pooling
        x = Flatten()(x)
        x = Dense(n_filters, activation='relu')(x)
        preds = Dense(len(self.category_map), activation='softmax')(x)

        # Compile model
        print('Compiling network...')
        self.model = Model(sequence_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['acc'])

    def train(self, data, labels, validation_split=0.2, batch_size=256, epochs=2):
        """Train classifier

        Args:
            data (np.array): 3D numpy array (n_samples, embedding_dim, tokenizer.max_sequence_length)
            labels (np.array): 2D numpy array (n_samples, len(self.category_map))
            validation_split (float): Fraction of samples to be used for validation
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
        """
        print('Training...')
        # Split the data into a training set and a validation set
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(validation_split * data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]

        # Train!
        self.save()
        checkpointer = ModelCheckpoint(filepath=self.prefix+'.h5', verbose=1, save_best_only=False)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val),
                       callbacks=[checkpointer],
                       epochs=epochs, batch_size=batch_size)
        self.evaluate(x_val, y_val, batch_size)

    def evaluate(self, x_test, y_test, batch_size=256):
        """Evaluate classifier

        Args:
            x_test (np.array): 3D numpy array (n_samples, embedding_dim, tokenizer.max_sequence_length)
            y_test (np.array): 2D numpy array (n_samples, len(self.category_map))
            batch_size (int): Training batch size
        """
        print('Evaluating...')
        predictions_last_epoch = self.model.predict(x_test, batch_size=batch_size, verbose=1)
        predicted_classes = np.argmax(predictions_last_epoch, axis=1)
        target_names = ['']*len(self.category_map)
        for category in self.category_map:
            target_names[self.category_map[category]] = category
        y_val = np.argmax(y_test, axis=1)
        print((classification_report(y_val, predicted_classes, target_names=target_names, digits = 6,labels=range(len(self.category_map)))))
