"""Script to train a product category ner based on product titles and descriptions
"""

import sys, csv
from tokenizer import WordTokenizer
from ner import ProductNER

MAX_TEXTS = 1000000

def usage():
    print("""
USAGE: python train_ner.py data_file.csv
FORMAT: "title","brand","description","categories"
""")
    sys.exit(0)

def main(argv):
    if len(argv) < 2:
        usage()

    # Fetch data
    texts, tags = [], []
    with open(sys.argv[1], 'r') as f:
        reader = csv.DictReader(f, fieldnames=["title","brand","description","categories","tags"])
        count = 0
        for row in reader:
            count += 1
            text, tag_set = row['title'], row['tags'].split(' ')[:-1]
            texts.append(text)
            tags.append(tag_set)
            if count >= MAX_TEXTS:
                break
    print(('Processed %s texts.' % len(texts)))

    # Tokenize texts
    tokenizer = WordTokenizer()
    tokenizer.load()
    data = tokenizer.tokenize(texts)

    # Get labels from NER
    ner = ProductNER()
    labels = ner.get_labels(tags)

    # Compile NER network and train
    ner.compile(tokenizer)
    ner.train(data, labels, epochs=2)

if __name__ == "__main__":
    main(sys.argv)
