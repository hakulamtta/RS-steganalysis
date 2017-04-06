# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 20:22:38 2017

@author: fanfan
"""

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import pylab as pl
import random  

#对像素块进行Z字排序
def Z(tmp):
    size = 8
    a = [0] * 64
    n = 0
    i = 0
    j = 0
    status = 0 #标记当前状态 0：右上运动 1：左下运动
    while (n<size*size):
        if (((i==0 and j%2!=0) or (j==1 and i==0)) and j!=size-1): #当前位于像素矩阵的最左边的奇数位且不是边界，向下移动,下一步向右上运动
            a[n] = tmp[i][j]
            j = j + 1
            n = n + 1
            status = 0    
        if (((i==0 and j%2!=0) or (j==1 and i==0)) and j==size-1): #当前位于像素矩阵的最左边的奇数位且是边界，向右移动,下一步向右上运动
            a[n] = tmp[i][j]
            i = i + 1
            n = n + 1
            status = 0    
        elif ((j==0 and i%2==0 and i>1) or (i==0 and j==0)): #当前位于像素矩阵的最上面的偶数位，向右移动，下一步向左下运动             
            a[n] = tmp[i][j]
            i = i + 1
            n = n + 1
            status = 1
        elif ((i==0 and j%2==0 and j>1) or (j==0 and i==0)): #当前位于像素矩阵的最左边的偶数位，向右上移动
            a[n] = tmp[i][j]
            j = j - 1
            i = i + 1
            n = n + 1
            status = 0
        elif ((j==0 and i%2!=0) or (i==1 and j==0)): #当前位于像素矩阵的最上面的奇数位，向左下移动
            a[n] = tmp[i][j]
            i = i - 1
            j = j + 1
            n = n + 1
            status = 1
        elif ((i==size-1 and j%2!=0)): #当前位于像素矩阵的最右边的奇数位，向下移动,下一步向左下运动
            a[n] = tmp[i][j]
            j = j + 1
            n = n + 1
            status = 1
        elif ((j==size-1 and i%2==0 and i>1)): #当前位于像素矩阵的最下面的偶数位，向右移动，下一步向右上移动              
            a[n] = tmp[i][j]
            i = i + 1
            n = n + 1
            status = 0
        elif ((i==size-1 and j%2==0 and j>1)): #当前位于像素矩阵的最右边的偶数位，向左下移动
            a[n] = tmp[i][j]
            i = i - 1
            j = j + 1
            n = n + 1
            status = 1
        elif ((j==size-1 and i%2!=0)): #当前位于像素矩阵的最下面的奇数位，向右上移动
            a[n] = tmp[i][j]
            i = i + 1
            j = j - 1
            n = n + 1
            status = 0
        else: #不是边界条件时，使用状态值判断移动方向
            if (status == 0): #右上运动
                a[n] = tmp[i][j]
                i = i + 1
                j = j - 1
                n = n + 1
                status = 0
            elif (status == 1): #左下运动
                a[n] = tmp[i][j]
                i = i - 1
                j = j + 1
                n = n + 1
                status = 1          
    return a

#计算像素相关性
def Calculate(a):
    res = 0
    for i in range(63):
        if (a[i+1] > a[i]):
            res = res + a[i+1] - a[i]
        else:
            res = res + a[i] - a[i+1]
    return res

#0翻转
def F0(val):  
    return val

#正翻转
def F1(val):
    if (val%2==0 and val!=1): #偶数加一
        val = val + 1
    elif (val%2==1 or val==1): #奇数减一
        val = val - 1
    return val
    
#负翻转
def F_1(val):
    if (val%2==0 and val!=1): #偶数减一
        val = val - 1
    elif (val%2==1 or val==1): #奇数加一
        val = val + 1
    return val
    
#生成随机数组
def Random(typ):
    ran = [0] * 64
    if (typ == 1):
        for i in range(64):
            ran[i] = random.randint(0,1)
    elif (typ == -1):
        for i in range(64):
            ran[i] = random.randint(-1,0)
    return ran

#RS隐写分析
def RS(tmp):
#==============================================================================
#     非负反转
#==============================================================================
    ran = Random(1)
    rev = [[0 for co in range(8)] for ro in range(8)] # 进行反转后的二维数组
    rm = 0
    sm = 0
    
    r1 = Z(tmp)
    res1 = Calculate(r1)#反转之前的像素相关性
    
    k = 0
    for i in range(8):
        for j in range(8):
            if (ran[k] == 0):#F0翻转
                rev[i][j] = F0(tmp[i][j])
            elif (ran[k] == 1):#F1翻转
                rev[i][j] = F1(tmp[i][j])
            k = k + 1
        
    r2 = Z(rev) # 将图像块进行Z字形排序
    res2 = Calculate(r2)#翻转之后的像素相关性
    
    if (res1 > res2):
        sm = sm + 1
    elif (res1 < res2):
        rm = rm + 1
        
#==============================================================================
#     非正翻转
#==============================================================================
    k = 0
    r_m = 0
    s_m = 0
    ran = Random(-1)
    rev = [[0 for co in range(8)] for ro in range(8)] # 进行反转后的二维数组
    for i in range(8):
        for j in range (8):
            if (ran[k] == 0):
                rev[i][j] = F0(tmp[i][j])
            elif (ran[k] == -1):
                rev[i][j] = F_1(tmp[i][j])
            k = k + 1
            
    r3 = Z(rev) # 将图像块进行Z字形排序
    res3 = Calculate(r3)#翻转之后的像素相关性
    
    if (res1 > res3):
        s_m = s_m + 1
    elif (res1 < res3):
        r_m = r_m + 1
    
    res = [rm,sm,r_m,s_m]
    return res

#LSB隐写
def LSB(fig,rate):
    s = int(512*512*rate)
    sec = [0] * (s)
    k = 0
    
    for i in range(s):
        sec[i] = random.randint(0,1)
    
#    print sec    
#    print fig
    
    for i in range(512):
        for j in range(512):
            if (k < s):
                if (sec[k]==1 and fig[i][j]%2==0 and fig[i][j]%2!=1):#偶数嵌入1
                    fig[i][j] = fig[i][j] + 1
                    k += 1
                elif ((sec[k]==1 and fig[i][j]%2==1) or fig[i][j]==1):#奇数嵌入1
                    fig[i][j] = fig[i][j] + 0
                    k += 1
                elif (sec[k]==0 and fig[i][j]%2==0 and fig[i][j]%2!=1):#偶数嵌入0
                    fig[i][j] = fig[i][j] + 0
                    k += 1
                elif ((sec[k]==0 and fig[i][j]%2==1) or fig[i][j]==1):#奇数嵌入0
                    fig[i][j] = fig[i][j] - 1
                    k += 1
    return fig
#==============================================================================
# 正经玩意儿
#==============================================================================
img = mpimg.imread('lena.bmp')
plt.imshow(img) # 显示图片
#plt.imshow(img_1,cmap='gray')
plt.axis('off') # 不显示坐标轴
plt.show()
print img.shape 

#将图片信息转存在二维数组中
img_1 = img[:,:,0]
fig = np.array(img_1)
#print fig

#对图像分块并计算相关性
tmp = [[0 for co in range(8)] for ro in range(8)]
#print len(tmp)
#print tmp
row = img.shape[0] / 8   #行
col = img.shape[1] / 8   #列
result = [0] *4
nn = 0

for i in range(64): 
    for j in range(64):
        x=0
        for k in range(i * 8, (i+1) * 8):
            y=0
            for h in range(j * 8, (j+1) * 8):
                tmp[x][y] = fig[k][h]
                y=y+1
            x=x+1
       # print tmp
        res = RS(tmp)# 进行RS隐写分析,res = [rm,sm,r_m,s_m]
        for n in range(4):
            result[n] = result[n] + res[n]
print 'orignal picture:'
print 'rm=',result[0],'  sm=',result[1],'  r_m=',result[2],'  s_m=',result[3]

#进行LSB隐写之后
rate = 0
rm = [0] * (11)
sm = [0] * (11)
r_m = [0] * (11)
s_m = [0] * (11)
while (rate <= 1.0):
    fig =  LSB(fig,rate)
    #对图像分块并计算相关性
    tmp = [[0 for co in range(8)] for ro in range(8)]
    #print len(tmp)
    #print tmp
    row = img.shape[0] / 8   #行
    col = img.shape[1] / 8   #列
    result = [0] *4
    
    for i in range(64): 
        for j in range(64):
            x=0
            for k in range(i * 8, (i+1) * 8):
                y=0
                for h in range(j * 8, (j+1) * 8):
                    tmp[x][y] = fig[k][h]
                    y=y+1
                x=x+1
           # print tmp
            res = RS(tmp)# 进行RS隐写分析,res = [rm,sm,r_m,s_m]
            for n in range(4):
                result[n] = result[n] + res[n]
    print 'rate=',rate,'after LSB:'
    print 'rm=',result[0],'  sm=',result[1],'  r_m=',result[2],'  s_m=',result[3]
    rate += 1
    

    rm[nn] = result[0]
    sm[nn] = result[1]
    r_m[nn] = result[2]
    s_m[nn] = result[3]
    nn = nn + 1
    
rate = 0
arr = [0] * 11
k = 0
while (k <= 10):
    rate = rate + 0.1
    arr[k] = rate
    k += 1
RM=pl.plot(arr,rm)
SM=pl.plot(arr,sm)
R_M=pl.plot(arr,r_m)
S_M=pl.plot(arr,s_m)
pl.show()