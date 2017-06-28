"""Normalizes product data"""

import sys, csv
import htmllib

def unescape(s):
    p = htmllib.HTMLParser(None)
    p.save_bgn()
    try:
        p.feed(s)
    except:
        return s
    return p.save_end()

in_file = sys.argv[1]
out_file = '.'.join(in_file.split('.')[:-1] + ['normalized'] + ['csv'])
with open(in_file, 'rb') as f:
    reader = csv.reader(f)
    writer = csv.writer(open(out_file,"w"))
    count = 0
    for row in reader:
        count += 1
        if not (count % 10000):
            print count, 'rows normalized'
        row = [unescape(x).lower().replace('\\n', ' ') for x in row]
        writer.writerow(row)
    print count, 'rows normalized'
