import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('data_statlog/sat.trn',sep=' ',encoding='utf-8')
print(len(data.iloc[:,-1]))

lop1= lop2= lop3= lop4= lop5= lop7=0 
for i in range(0,len(data.iloc[:,-1])):
    if(data.iloc[i,-1] == 1):
        lop1+=1
    if(data.iloc[i,-1] == 2):
        lop2+=1
    if(data.iloc[i,-1] == 3):
        lop3+=1
    if(data.iloc[i,-1] == 4):
        lop4+=1
    if(data.iloc[i,-1] == 5):
        lop5+=1
    if(data.iloc[i,-1] == 7):
        lop7+=1

print(f'Lop1: {lop1}')
print(f'Lop2: {lop2}')
print(f'Lop3: {lop3}')
print(f'Lop4: {lop4}')
print(f'Lop5: {lop5}')
print(f'Lop7: {lop7}')
print(f'Tong: {lop1+lop2+lop3+lop4+lop5+lop7} - len:{len(data.iloc[:,-1])}')


Nhan = ['red soil','cotton crop','grey soil','damp grey soil',
        'soil with vegetation stubble','very damp grey soil']
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
SoLop = [lop1,lop2,lop3,lop4,lop5,lop7]

ax.pie(SoLop,labels=Nhan ,autopct='%1.2f%%')
plt.legend(loc = 'upper right',) #thêm chú thích
plt.show()