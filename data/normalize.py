"""Normalizes product data"""

import sys, csv

def unescape(s):
    if sys.version_info >= (3, 0):
        import html
        output = html.unescape(str(s))
    else:
        import htmllib

        p = htmllib.HTMLParser(None)
        p.save_bgn()
        try:
            p.feed(s)
        except:
            return s
        output=p.save_end()
    return output


in_file = sys.argv[1]
out_file = '.'.join(in_file.split('.')[:-1] + ['normalized'] + ['csv'])
with open(in_file, 'r') as f:
    reader = csv.reader(f)
    writer = csv.writer(open(out_file,"w"))
    count = 0
    for row in reader:
        count += 1
        if not (count % 10000):
            print (count, 'rows normalized')
        row = [unescape(x).lower().replace('\\n', ' ') for x in row]
        writer.writerow(row)
    print (count, 'rows normalized')
