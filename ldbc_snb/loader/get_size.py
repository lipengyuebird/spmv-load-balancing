# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2022/12/23 15:00
# @Author   : Perye(Li Pengyu)
# @FileName : get_size.py
# @Software : PyCharm
import json

s = set()

with open('sf10_dataset/Person_knows_Person.json', 'r') as f_pkp:
    pkp_idx = json.load(f_pkp)
    for i in pkp_idx:
        for j in i:
            s.add(j)


print(len(s))

with open('sf10_dataset/Person_knows_Person1.json', 'r') as f_pkp:
    pkp_idx = json.load(f_pkp)

print(max(max(pkp_idx[0]), max(pkp_idx[1])) + 1)
