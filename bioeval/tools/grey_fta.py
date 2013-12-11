#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as N
import sqlite3

"""
Get the Failure to Acquire information
from the Greyc Keystroke Database
AUTHOR Romain Giot <romain.giot@ensicaen.fr>
"""

OUTPUT= '../data/GREYC_FT.txt'
INPUT= '../../PythonKDA/data/greyc-keystroke-alpha.db'
QUERY =  "select user_id, success from keystroke_typing"

users = {}
res = []

#Launch reading procedure
conn = sqlite3.connect(INPUT)
c = conn.cursor()
c.execute(QUERY)
for row in c:
    user = int(row[0])
    failure = not bool(row[1])

    if user not in users:
        users[user] = 1
    else:
        users[user] = users[user] + 1

    line = N.array( (user, users[user], failure) )
    res.append(line)

#Store in file
N.savetxt( OUTPUT, res)

#Verify
res = N.loadtxt(OUTPUT)
print res
