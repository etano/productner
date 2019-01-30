"""Supplements product data"""

import sys, csv

in_file = sys.argv[1]
out_file = '.'.join(in_file.split('.')[:-1] + ['supplemented'] + ['csv'])
with open(in_file, 'r') as f:
    reader = csv.reader(f)
    writer = csv.writer(open(out_file,"w"))
    count, supplemented = 0, 0
    for row in reader:
        count += 1
        if not (count % 10000):
            print (supplemented, '/', count, 'rows supplemented')
        title, brand, description = row[0], row[1], row[2]
        if not (brand in title):
            supplemented += 1
            title = brand + ' ' + title
        description = title + ' ' + description
        row[0], row[1], row[2] = title, brand, description
        writer.writerow(row)
    print (supplemented, '/', count, 'rows supplemented')
