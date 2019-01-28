import sys, csv
from operator import itemgetter

with open(sys.argv[1], 'r') as f:
    reader = csv.reader(f)
    brands, categories = {}, {}
    count = 0
    for row in reader:
        count += 1
        if not (count % 10000): print(count)
        brand = row[1]
        if brand in brands: brands[brand] += 1
        else: brands[brand] = 1
        category = row[3].split(' / ')[0]
        if category in categories: categories[category] += 1
        else: categories[category] = 1
    print( sorted(brands.items(), key=itemgetter(1)))
    print( sorted(categories.items(), key=itemgetter(1)))
