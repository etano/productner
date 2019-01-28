"""Named entity recognition class"""

import os, json
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, Activation
from keras.models import load_model, Sequential
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

class ProductNER(object):
    """Class which recognizes named entities

       Attributes:
           prefix (str): Model files prefix
           model (keras.model): Keras model
           tag_map (dict(str, int)): Map between tag names and indices
    """

    def __init__(self, prefix=None):
        """Load in model and tag map

        Args:
            prefix (str): Prefix of directory containing model HDF5 file and tag map JSON file
        """
        if prefix != None:
            self.load(prefix)
        else:
            self.prefix = 'models/ner'
            self.model = None
            self.tag_map = {}

    def load(self, prefix=None):
        """Load in model and tag map

        Args:
            prefix (str): Prefix of directory containing model HDF5 file and tag map JSON file
        """
        if prefix != None: self.prefix = prefix
        self.model = load_model(self.prefix+'.h5')
        self.tag_map = json.load(open(self.prefix+'.json', 'r'))

    def save(self, prefix=None):
        """Save in model and tag map

        Args:
            prefix (str): Prefix of directory containing model HDF5 file and tag map JSON file
        """
        if prefix != None: self.prefix = prefix
        self.model.save(self.prefix+'.h5')
        with open(self.prefix+'.json', 'w') as out:
            json.dump(self.tag_map, out)

    def tag(self, data):
        """Return all named entities given some embedded text

        Args:
            data (np.array): 2D array representing descriptions of the product and/or product title
        Returns:
            list(list(dict(str, float))): List of lists of entities
        """
        prediction = self.model.predict(data)
        all_tag_probs = []
        for i in range(prediction.shape[0]):
            sentence_tag_probs = []
            first_word = 0
            for j in range(data[i].shape[0]):
                if data[i,j] != 0: break
                first_word += 1
            for j in range(first_word, prediction.shape[1]):
                word_tag_probs = {}
                for tag in self.tag_map:
                    word_tag_probs[tag] = prediction[i,j,self.tag_map[tag]]
                sentence_tag_probs.append(word_tag_probs)
            all_tag_probs.append(sentence_tag_probs)
        return all_tag_probs

    def index_tags(self, tags):
        """Take a list of possibly duplicate tags and create an index list

        Args:
            tags (list(str)): List of tags
        Returns:
            list(int): List of indices
        """
        indices = []
        for tag in tags:
            if not (tag in self.tag_map):
                self.tag_map[tag] = len(self.tag_map) + 1
            indices.append(self.tag_map[tag])
        return indices

    def get_labels(self, tag_sets):
        """Create labels from a list of tag_sets

        Args:
            tag_sets (list(list(str))): A list of word tag sets
        Returns:
            (list(list(int))): List of list of indices
        """
        labels = []
        print('Getting labels...')
        for tag_set in tag_sets:
            indexed_tags = self.index_tags(tag_set)
            labels.append(to_categorical(np.asarray(indexed_tags), num_classes=4))
        labels = pad_sequences(labels, maxlen=200)
        return labels

    def compile(self, tokenizer, glove_dir='./data/', embedding_dim=200, dropout_fraction=0.2, hidden_dim=32):
        """Compile network model for NER

        Args:
            glove_file (str): Location of GloVe file
            embedding_dim (int): Size of embedding vector
            tokenizer (WordTokenizer): Object used to tokenize orginal texts
            dropout_fraction (float): Fraction of randomly zeroed weights in dropout layer
            hidden_dim (int): Hidden dimension
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

        # Create network
        print('Creating network...')
        self.model = Sequential()
        self.model.add(Embedding(len(tokenizer.tokenizer.word_index) + 1,
                                 embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=tokenizer.max_sequence_length,
                                 trainable=False,
                                 mask_zero=True))
        self.model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(len(self.tag_map) + 1)))
        self.model.add(Activation('softmax'))

        # Compile model
        print('Compiling network...')
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['acc'])

    def train(self, data, labels, validation_split=0.2, batch_size=256, epochs=2):
        """Train ner

        Args:
            data (np.array): 3D numpy array (n_samples, embedding_dim, tokenizer.max_sequence_length)
            labels (np.array): 3D numpy array (n_samples, tokenizer.max_sequence_length, len(self.tag_map))
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

        print(data.shape, labels.shape)

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
            x_test (np.array): 2D numpy array (n_samples, tokenizer.max_sequence_length)
            y_test (np.array): 3D numpy array (n_samples, tokenizer.max_sequence_length, len(self.tag_map))
            batch_size (int): Training batch size
        """
        print('Evaluating...')
        predictions_last_epoch = self.model.predict(x_test, batch_size=batch_size, verbose=1)
        predicted_classes = np.argmax(predictions_last_epoch, axis=2).flatten()
        y_val = np.argmax(y_test, axis=2).flatten()
        target_names = ['']*(max(self.tag_map.values())+1)
        for category in self.tag_map:
            target_names[self.tag_map[category]] = category

        print((classification_report(y_val, predicted_classes, target_names=target_names, digits = 6, labels=range(len(target_names)))))
