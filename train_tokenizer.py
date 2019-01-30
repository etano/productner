"""Script to train a word tokenizer
"""

import sys, csv
from tokenizer import WordTokenizer

MAX_TEXTS = 1000000

def usage():
    print("""
USAGE: python train_tokenizer.py data_file.csv
FORMAT: "title","brand","description","categories"
""")
    sys.exit(0)

def main(argv):
    if len(argv) < 2:
        usage()

    # Fetch data
    texts, categories = [], []
    with open(sys.argv[1], 'r') as f:
        reader = csv.DictReader(f, fieldnames=["title","brand","description","categories"])
        count = 0
        for row in reader:
            count += 1
            text, category = row['title']+' '+row['description'], row['categories'].split(' / ')[0]
            texts.append(text)
            categories.append(category)
            if count >= MAX_TEXTS:
                break
    print(('Processed %s texts.' % len(texts)))

    # Tokenize texts
    tokenizer = WordTokenizer()
    tokenizer.train(texts)

if __name__ == "__main__":
    main(sys.argv)
