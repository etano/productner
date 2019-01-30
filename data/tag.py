"""Tags product data"""

import sys, csv

in_file = sys.argv[1]
out_file = '.'.join(in_file.split('.')[:-1] + ['tagged'] + ['csv'])
with open(in_file, 'r') as f:
    reader = csv.reader(f)
    writer = csv.writer(open(out_file,"w"))
    count = 0
    for row in reader:
        count += 1
        if not (count % 10000):
            print (count, 'rows tagged')
        title, brand, description = row[0], row[1], row[2]
        tagging = ''
        brand = brand.split(' ')
        brand_started = False
        for word in title.split(' '):
            if word == brand[0]:
                tagging += 'B-B '
                brand_started = True
            elif len(brand) > 1 and brand_started:
                for b in brand[1:]:
                    if word == b:
                        tagging += 'I-B '
                    else:
                        brand_started = False
                        tagging += 'O '
            else:
                brand_started = False
                tagging += 'O '
        row.append(tagging)
        writer.writerow(row)
    print (count, 'rows tagged')
