# Product categorization and named entity recognition

This repository is meant to automatically extract features from product titles and descriptions. Below we explain how to install and run the code, and the implemented algorithms. We also provide background information including the current state-of-the-art in both sequence classification and sequence tagging, and suggest possible improvements to the current implemention. Enjoy!

## Requirements

The following libraries are required:

- Keras >2.0 (https://keras.io/#installation)
- Tensorflow >r1.0 (https://www.tensorflow.org/install/)
- SKlearn >0.18 (http://scikit-learn.org/stable/) (for metrics only)

## Usage

### Fetching data

#### Amazon product data

    cd ./data/
    wget http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
    gzip -d metadata.json.gz

#### GloVe

    cd ./data/
    wget http://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip

### Preprocessing data

    cd ./data/
    python parse.py metadata.json
    python normalize.py products.csv
    python trim.py products.normalized.csv
    python supplement.py products.normalized.trimmed.csv
    python tag.py products.normalized.trimmed.supplemented.csv

### Training models

    mkdir -p ./models/
    python train_tokenizer.py data/products.normalized.trimmed.supplemented.tagged.csv
    python train_classifier.py data/products.normalized.trimmed.supplemented.tagged.csv
    python train_ner.py data/products.normalized.trimmed.supplemented.tagged.csv

### Extract information

Infer on our sample dataset with your model by running the following:

    python extract.py ./models/ Product\ Dataset.csv

## Contents

- extract.py: Script to extract product category specific attributes based on product titles and descriptions
- train_tokenizer.py: Script to train a word tokenizer
- train_ner.py: Script to train a product named entity recognizer based on product titles
- train_classifier.py: Script to train a product category classifier based on product titles and descriptions
- tokenizer.py: Word tokenizer class
- ner.py: Named entity recognition class
- classifier.py: Product classifier class
- data/parse.py: Parses Amazon product metadata found at http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
- data/normalize.py: Normalizes product data
- data/trim.py: Trims product data
- data/supplement.py: Supplements product data
- data/tag.py: Tags product data
- Product\ Dataset.csv: CSV file with product ids, names, and descriptions

## Algorithms

These are the methods used in this demonstrative implementation. For state-of-the-art extensions, we refer the reader to the references listed below.

- Tokenization: Built-in Keras tokenizer with 80,000 word maximum
- Embedding: Stanford GloVe (Wikipedia 2014 + Gigaword 5, 200 dimensions) with 200 sequence length maximum
- Sequence classification: 3 layer CNN with max pooling between the layers
- Sequence tagging: Bidirectional LSTM

For the sequence classification task, we extract product titles, descriptions, and categories from the Amazon product corpus. We then fit our CNN model to predict product category based on a combination of product title and description. On 800K samples with a batch size of 256, we achieve an overall f1 score of ~0.90 after 2 epochs.

For the sequence tagging task, we extract product titles and brands from the Amazon product corpus. We then fit our bidirection LSTM model to label each word token in the product title to be either a brand or not. On 800K samples with a batch size of 256, we achieve an overall f1 score of ~0.85 after 2 epochs.

For both models we use the GloVe embedding with 200 dimensions, though we note that a larger dimensional embedding might achieve superior performance. Additionally, we could be more careful in the data preprocessing to trim bad tokens (e.g. HTML remnants). Also for both models we use a dropout layer after embedding to combat overfitting the data.

## Background

### Problem definition

The problem of extracting features from unstructured textual data can be given different names depending on the circumstances and desired outcome. Generally, we can split tasks into two camps: sequence classification and sequence tagging.

In sequence classification, we take a text fragment (usually a sentence up to an entire document), and try to project it into a categorical space. This is considered a many-to-one classification in that we are taking a set of many features and producing a single output.

Sequence tagging, on the other hand, is often considered a many-to-many problem since you take in an entire sequence and attempt to apply a label to each element of the sequence. An example of sequence tagging is part of speech labeling, where one attempts to label the part of speech of each word in a sentence. Other methods that fall into this camp include chunking (breaking a sentence into relational components) and named entity recognition (extracting pre-specified features like geographic locations or proper names).

### Tokenization and embedding

An often important step in any natural language processing task is projecting from the character-based space that composes words and sentences to a numeric space on which computer models can operate.

The first step is simply to index unique tokens appearing in a dataset. There is some freedom on what is considered a token, i.e. it can be considered a specific group of words, a single word, or even individual characters. A popular choice is to simple create a word-based dictionary which maps unique space-separated character sequences to unique indices. Usually this is done after a normalization procedure where everything is lower-cased, made into ASCII, etc. This dictionary can then be sorted by frequency of occurance in the dataset and truncated to a maximum size. After tokenization, your dataset is transformed into a set of indices where truncated words are typically replaced with a '0' index.

Following tokenization, the indexed words are often projected into an embedding vector space. Currently popular embeddings include word2vec [\[1\]](#references) and GloVe [\[2\]](#references). Word2vec (as the name implies) is a word to vector space projector composed of a two-layer neural network. The network is trained in one of two ways: a continuous bag-of-words where the model attempts to predict the current word by using the surrounding words as context features, and continuous skip-grams where the model attempts to predict surrounding context words by looking at the current word. GloVe is a "global vectors representation for words". Essentially it is a count-based, unsupervised learned embedding where a token cooccurance matrix is constructed and factored.

Vector-space embedding methods generally provide substantial improvements over using basic dictionaries since they inject contextual knowledge from the language. Additionally, they allow a much more compact representation, while maintaining important correlations. For example, they allow you to do amazing things like performing word arithmetic:

    king - man + woman = queen

where equality is determined by directly computing vector overlaps.

### Sequence classification

Classification is the shining pillar of modern day machine learning with convolutional neural networks (CNN) at the top. With their ability to efficiently represent high-level features via windowed filtering, CNN's have seen their largest success in the classification and segmentation of images. However, more recently, CNN's have started seeing success in natural language sequence classification as well. Several recent works have shown that for the text classification, CNN's can significantly outperform other classifying methods such as hidden Markov models and support vector machines [\[3,4\]](#references). The reason CNN's see success in text classification is likely for the same reason they see success in the vision domain: there are strong, regular correlations between nearby features which are efficiently picked up by reasonably sized filters.

Even more recently CNN's dominance has been toppled by the recurrent neural network (RNN) architectures. In particular, long-/short-term memory (LSTM) units have shown exceptional promise. LSTM's pass output from one unit to the next, while carrying along an internal state. How this state updates (as well as other weights in the network) can be trained end-to-end on variable length sequences by passing a single token at a time. For classification, bidirectional LSTM's, which allow for long-range contextual correlations in both forward and reverse directions, have seen the best performance [\[5,6\]](#references). An additional feature of these networks is an attention layer that allows continuous addressing of internal states of the sequential LSTM units. This further strengthens the networks ability to draw correlations from both nearby and far away tokens.

### Sequence tagging

As mentioned above sequence tagging is a many-to-many machine learning task, and thus an added emphasis on the sequential nature of the input and output. This makes largely CNN's ill-suited for the problem. Instead the dominant approaches are again bidirectional LSTM's [\[11,12\]](#references) as well as another method called conditional random fields (CRF) [\[7\]](#references). CRF's can be seen as either sequential logistic regression or more powerful hidden Markov models. Essentially they are sequential models composed of many defined feature functions that depend both on the word currently be labelled as well as surrounding words. The relative weights of these feature functions can then be trained via any supervised learning approach. CRF's are used extensively in the literature for both part of speech tagging as well as named entity recognition because of their ease of use and intuitive feeling [\[8-10\]](#references).

Even more recent models for sequence tagging use a combination of the aforementioned methods (CNN, LSTM, and CRF) [\[13,14,15\]](#references).  These works usually use a bidirectional LSTM as the major labeling architecture, another RNN or CNN to capture character-level information, and finally a CRF layer to model the label dependency. A logical next step will be to combine these methods with the neural attention models used in sequence classification, though this seems to be currently missing from the literature.

### Future directions

Looking forward, there are several available avenues for continued research. More sophisticated word embeddings might help alleviate the need for complicated neural architectures. Hierarchical optimization methods can be used to automatically build new architectures as well as optimize hyperparameters. Diverse models can be intelligently combined to produce more powerful classification schemes (indeed most all Kaggle competitions are won this way). One interesting approach is to combine text data with other available data sources such as associated images [\[10\]](#references). By collecting data from different sources, feature labels could possibly be extracted automatically by cross-comparison.

## References

[1] "Distributed Representations of Words and Phrases and their Compositionality". Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean. 2013. https://arxiv.org/abs/1310.4546.

[2] "GloVe: Global Vector Representation for Words". Stanford NLP. 2015. https://nlp.stanford.edu/projects/glove/.

[3] "Convolutional Neural Networks for Sentence Classification". Yoon Kim. 2014. "https://arxiv.org/abs/1408.5882.

[4] "Character-level Convolutional Networks for Text Classification". Xiang Zhang, Junbo Zhao, Yann LeCun. 2015. https://arxiv.org/abs/1509.01626.

[5] "Document Modeling with Gated Recurrent Neural Network for Sentiment Classification". Duyu Tang, Bing Qin, Ting Liu. 2015. http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP167.pdf.

[6] "Hierarchical Attention Networks for Document Classification". Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, Eduard Hovy. 2016. https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf.

[7] "An Introduction to Conditional Random Fields". Charles Sutton, Andrew McCallum. 2010. https://arxiv.org/abs/1011.4088.

[8] "Attribute Extraction from Product Titles in eCommerce". Ajinkya More. 2016. https://arxiv.org/abs/1608.04670.

[9] "Bootstrapped Named Entity Recognition for Product Attribute Extraction". Duangmanee (Pew) Putthividhya, Junling Hu. 2011. http://www.aclweb.org/anthology/D11-1144.

[10] "A Machine Learning Approach for Product Matching and Categorization". Petar Ristoski, Petar Petrovski, Peter Mika, Heiko Paulheim. 2017. http://www.semantic-web-journal.net/content/machine-learning-approach-product-matching-and-categorization-0.

[11] "Multilingual Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Models and Auxiliary Loss". Barbara Plank, Anders Søgaard, Yoav Goldberg. 2016. https://arxiv.org/abs/1604.05529.

[12] "Part-of-Speech Tagging with Bidirectional Long Short-Term Memory Recurrent Neural Network". Peilu Wang, Yao Qian, Frank K. Soong, Lei He, Hai Zhao. 2015. https://arxiv.org/abs/1510.06168.

[13] "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF". Xuezhe Ma, Eduard Hovy. 2016. https://arxiv.org/abs/1603.01354.

[14] "Neural Architectures for Named Entity Recognition". Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. 2016. https://arxiv.org/abs/1603.01360.

[15] "Neural Models for Sequence Chunking". Feifei Zhai, Saloni Potdar, Bing Xiang, Bowen Zhou. 2017. https://arxiv.org/abs/1701.04027.
