"""Trims product data"""

import sys, csv

in_file = sys.argv[1]
out_file = '.'.join(in_file.split('.')[:-1] + ['trimmed'] + ['csv'])
with open(in_file, 'r') as f:
    reader = csv.reader(f)
    writer = csv.writer(open(out_file,"w"))
    count, trimmed = 0, 0
    for row in reader:
        try:
            count += 1
            if not (count % 10000):
                print (trimmed, '/', count, 'rows trimmed')
            brand = row[1].lower()
            if brand == 'unknown' or brand == '' or brand == 'generic':
                trimmed += 1
                continue
            writer.writerow(row)
        except:
            print (row)
    print (trimmed, '/', count, 'rows trimmed')
