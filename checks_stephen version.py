
import numpy as np
import json
import pylab as pl

f = open('ST_spec_prob.data')
a = json.load(f)
f.close()

f = open('LT_spec_prob.data')
b = json.load(f)
f.close()

f = open('Noise_spec_prob.data')
c = json.load(f)
f.close()

pl.ion()

#a = f.read().splitlines()
#c = a[0].split(",")

# a[i][0] is filename, [1] is the species label, [2] is the np array
ST = []
count = 0
st = []
while count<len(a):
    file = a[count][0]
    while count<len(a) and a[count][0] == file:
        st.append(a[count][2][1])
        count+=1
    ST.append([file,st])
    st = []

for i in range(len(ST)):
    stmax = max(ST[i][1])
    stmean = 0
    ind = np.array(ST[i][1]).argsort()[-3:][::-1]
    for j in ind:
        stmean+=ST[i][1][j]
    stmean /= 3
    print(ST[i][0],stmax,stmean,np.mean(ST[i][1]))

LT = []
count = 0
lt = []
while count<len(b):
    file = b[count][0]
    while count<len(b) and b[count][0] == file:
        lt.append(b[count][2][0])
        count+=1
    LT.append([file,lt])
    lt = []

for i in range(len(LT)):
    ltmax = max(LT[i][1])
    ltmean = 0
    ind = np.array(LT[i][1]).argsort()[-3:][::-1]
    for j in ind:
        ltmean+=LT[i][1][j]
    ltmean /= 3
    print(LT[i][0],ltmax,ltmean,np.mean(LT[i][1]))

NT = []
count = 0
nt = []
while count<len(c):
    file = c[count][0]
    while count<len(c) and (c[count][0] == file or c[count][0] == 0):
        if c[count][0] != 0:
            nt.append(c[count][2][2])
        count+=1
    NT.append([file,nt])
    nt = []

for i in range(len(NT)):
    ntmax = max(NT[i][1])
    ntmean = 0
    ind = np.array(NT[i][1]).argsort()[-3:][::-1]
    for j in ind:
        ntmean+=NT[i][1][j]
    ntmean /= 3
    print(NT[i][0],ntmax,ntmean,np.mean(NT[i][1]))

