import sqlite3
import sys
import codecs


outfile = codecs.open(sys.argv[2], 'w', 'utf-8')
conn = sqlite3.connect(sys.argv[1])
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table';")
print('Tables:', c.fetchall())

for row in c.execute('SELECT * from store'):
    if row[0] == '\t':
        continue
    outfile.write(' '.join(str(x) for x in row[:-1]) + '\n')
