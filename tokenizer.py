"""Word tokenizer class"""

import os
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class WordTokenizer(object):
    """Class which tokenizes words

    Attributes:
        max_sequence_length (int): Maximum sequence length for embedding
        tokenizer (Tokenizer): Keras Tokenizer
        prefix (str): Prefix for tokenizer save file
    """

    def __init__(self, max_sequence_length=200, prefix="./models/tokenizer"):
        """Create tokenizer

        Args:
            max_sequence_length (int): Maximum sequence length for texts
            prefix (str): Prefix for tokenizer save file
        """
        self.max_sequence_length = max_sequence_length
        self.prefix = prefix
        self.tokenizer = None

    def save(self, prefix=None):
        """Saves the tokenizer

        Args:
            prefix (str): Prefix for tokenizer save file
        """
        if prefix != None: self.prefix = prefix
        pickle.dump(self.tokenizer, open(self.prefix+".pickle", "wb"))

    def load(self, prefix=None):
        """Loads the tokenizer
        """
        if prefix != None: self.prefix = prefix
        self.tokenizer = pickle.load(open(self.prefix+".pickle", "rb"))

    def train(self, texts, max_nb_words=80000):
        """Takes a list of texts, fits a tokenizer to them, and creates the embedding matrix.

        Args:
            texts (list(str)): List of texts
            max_nb_words: Maximum number of words indexed (take most frequently used)
        """
        # Tokenize
        print('Training tokenizer...')
        self.tokenizer = Tokenizer(num_words=max_nb_words)
        self.tokenizer.fit_on_texts(texts)
        self.save()
        print(('Found %s unique tokens.' % len(self.tokenizer.word_index)))

    def tokenize(self, texts):
        """Takes a list of texts and tokenizes them.

        Args:
            texts (list(str)): List of texts
        Returns:
            np.array: 2D numpy array (len(texts), self.max_sequence_length)
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return data
