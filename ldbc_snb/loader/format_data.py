# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/23 14:15
# @Author   : Perye(Li Pengyu)
# @FileName : format_data.py
# @Software : PyCharm

import json
import os

file_list = []
lines = []

for i in os.listdir('/home/perye/out-sf10/graphs/csv/bi/composite-merged-fk/initial_snapshot/dynamic/Person_knows_Person/'):
    if i.startswith('part'):
        file_list.append('/home/perye/out-sf10/graphs/csv/bi/composite-merged-fk/initial_snapshot/dynamic/Person_knows_Person/' + i)


for file in file_list:
    with open(file, 'r') as f:
        f.readline()
        lines.extend([list(map(int, line.strip().split("|")[1:])) for line in f.readlines()])


l = list(map(list, zip(*sorted(lines, key=lambda x: x[0]))))

counter = 0
d = {}

for i in l:
    for j in i:
        if j not in d.keys():
            d[j] = counter
            counter += 1


for i in range(2):
    for j in range(len(l[0])):
        l[i][j] = d[l[i][j]]


with open('sf10_dataset/Person_knows_Person1.json', 'w') as f:
    json.dump(l, f)
