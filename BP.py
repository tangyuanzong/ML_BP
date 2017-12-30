#!/usr/bin/python
# coding=utf-8
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import time
import xlrd
import sys


reload(sys)
sys.setdefaultencoding('utf-8')
matplotlib.rcParams['font.sans-serif']=['Droid Sans Fallback']
 
def read_xls_file(filename):                         #读取训练数据  
    data = xlrd.open_workbook(filename)                
    sheet1 = data.sheet_by_index(0)                    
    m = sheet1.nrows                                    
    n = sheet1.ncols                                    
    pop = []                         
    veh = []
    roa = []        
    pas = []
    fre = []
    for i in range(m):                                  
        row_data = sheet1.row_values(i)               
        if i > 0:
           pop.append(row_data[1])
           veh.append(row_data[2])
           roa.append(row_data[3])
           pas.append(row_data[4])
           fre.append(row_data[5])

    dataMat = np.mat([pop,veh,roa])
    labels = np.mat([pas,fre])
    dataMat_old = dataMat
    labels_old = labels
    return dataMat,labels,dataMat_old,labels_old

def read_xls_testfile(filename):                           #读取测试数据
    data = xlrd.open_workbook(filename) 
    sheet1 = data.sheet_by_index(0)            
    m = sheet1.nrows                           
    n = sheet1.ncols                                    
    pop = []                         
    veh = []
    roa = []        
    for i in range(m):                       
        row_data = sheet1.row_values(i)       
        if i > 0:
           pop.append(row_data[1])
           veh.append(row_data[2])
           roa.append(row_data[3])

    dataMat = np.mat([pop,veh,roa])
    return dataMat

def Norm(dataMat,labels):                                  #归一化数据
    dataMat_minmax = np.array([dataMat.min(axis=1).T.tolist()[0],dataMat.max(axis=1).T.tolist()[0]]).transpose() 
    dataMat_Norm = ((np.array(dataMat.T)-dataMat_minmax.transpose()[0])/(dataMat_minmax.transpose()[1]-dataMat_minmax.transpose()[0])).transpose()
    labels_minmax  = np.array([labels.min(axis=1).T.tolist()[0],labels.max(axis=1).T.tolist()[0]]).transpose()
    labels_Norm = ((np.array(labels.T).astype(float)-labels_minmax.transpose()[0])/(labels_minmax.transpose()[1]-labels_minmax.transpose()[0])).transpose()
    return dataMat_Norm,labels_Norm,dataMat_minmax,labels_minmax

def f(x):                                                  #激励函数
    return 1/(1+np.exp(-x))

def BP(sampleinnorm, sampleoutnorm):                       #BP算法
    maxepochs = 60000
    learnrate = 0.030
    errorfinal = 0.65*10**(-3)
    samnum = 20
    indim = 3
    outdim = 2
    hiddenunitnum = 3
    n,m = shape(sampleinnorm)
    w1 = 0.5*np.random.rand(hiddenunitnum,indim)-0.1
    b1 = 0.5*np.random.rand(hiddenunitnum,1)-0.1
    w2 = 0.5*np.random.rand(outdim,hiddenunitnum)-0.1
    b2 = 0.5*np.random.rand(outdim,1)-0.1

    errhistory = []

    for i in range(maxepochs):
        hiddenout = f((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
        networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
        err = sampleoutnorm - networkout
        sse = sum(sum(err**2))/m
        errhistory.append(sse)

        #if sse < errorfinal:
         #  break

        delta2 = err
        delta1 = np.dot(w2.transpose(),delta2)*hiddenout*(1-hiddenout)
        dw2 = np.dot(delta2,hiddenout.transpose())
        db2 = np.dot(delta2,np.ones((samnum,1)))
        dw1 = np.dot(delta1,sampleinnorm.transpose())
        db1 = np.dot(delta1,np.ones((samnum,1)))
        w2 += learnrate*dw2
        b2 += learnrate*db2
        w1 += learnrate*dw1
        b1 += learnrate*db1

    return errhistory,b1,b2,w1,w2,maxepochs


def show(sampleinnorm,sampleoutminmax,sampleout,errhistory,maxepochs):   #图形显示
    hiddenout = f((np.dot(w1,sampleinnorm).transpose()+b1.transpose())).transpose()
    networkout = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
    diff = sampleoutminmax[:,1]-sampleoutminmax[:,0]
    networkout2 = networkout
    networkout2[0] = networkout2[0]*diff[0]+sampleoutminmax[0][0]
    networkout2[1] = networkout2[1]*diff[1]+sampleoutminmax[1][0]
    sampleout = np.array(sampleout)

    fig,axes = plt.subplots(nrows=2,ncols=1,figsize=(12,10))
    line1, = axes[0].plot(networkout2[0],'k',markeredgecolor='b',marker = 'o',markersize=9)
    line2, = axes[0].plot(sampleout[0],'r',markeredgecolor='g',marker = u'$\star$',markersize=9)
    line3, = axes[0].plot(networkout2[1],'g',markeredgecolor='g',marker = 'o',markersize=9)
    line4, = axes[0].plot(sampleout[1],'y',markeredgecolor='b',marker = u'$\star$',markersize=9)
    axes[0].legend((line1,line2,line3,line4),(u'客运量预测输出',u'客运量真实输出',u'货运量预测输出',u'货运量真实输出'),loc = 'upper left')
    axes[0].set_ylabel(u'公路客运量及货运量')
    xticks = range(0,22,1)
    xtickslabel = range(1990,2012,1)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xtickslabel)
    axes[0].set_xlabel(u'年份')
    axes[0].set_title(u'BP神经网络')

    errhistory10 = np.log10(errhistory)
    minerr = min(errhistory10)
    plt.plot(errhistory10)
    axes[1]=plt.gca()
    axes[1].set_yticks([-2,-1,0,1,2,minerr])
    axes[1].set_yticklabels([u'$10^{-2}$',u'$10^{-1}$',u'$1$',u'$10^{1}$',u'$10^{2}$',str(('%.4f'%np.power(10,minerr)))])
    axes[1].set_xlabel(u'训练次数')
    axes[1].set_ylabel(u'误差')
    axes[1].set_title(u'误差曲线')

    plt.show()
    plt.close()
    
    return diff, sampleoutminmax

def pre(dataMat,dataMat_minmax,diff,sampleoutminmax,w1,b1,w2,b2):          #数值预测
    dataMat_test = ((np.array(dataMat.T)-dataMat_minmax.transpose()[0])/(dataMat_minmax.transpose()[1]-dataMat_minmax.transpose()[0])).transpose() 
    hiddenout = f((np.dot(w1,dataMat_test).transpose()+b1.transpose())).transpose()
    networkout1 = (np.dot(w2,hiddenout).transpose()+b2.transpose()).transpose()
    networkout = networkout1
    networkout[0] = networkout[0]*diff[0] + sampleoutminmax[0][0]
    networkout[1] = networkout[1]*diff[1] + sampleoutminmax[1][0]

    print "2010年预测的公路客运量为：", int(networkout[0][0]),"(万人)"
    print "2010年预测的公路货运量为：", int(networkout[1][0]),"(万吨)"
    print "2011年预测的公路客运量为：", int(networkout[0][1]),"(万人)"
    print "2011年预测的公路货运量为：", int(networkout[1][1]),"(万吨)"

dataMat,labels,dataMat_old,labels_old = read_xls_file('ex5data.xlsx')
dataMat_Norm,labels_Norm, dataMat_minmax, labels_minmax = Norm(dataMat,labels)
err, b1, b2, w1, w2,maxepochs = BP(dataMat_Norm,labels_Norm)
dataMat_test = read_xls_testfile('ex5data_test.xlsx')
diff, sampleoutminmax = show(dataMat_Norm,labels_minmax,labels,err,maxepochs)
pre(dataMat_test,dataMat_minmax,diff, sampleoutminmax ,w1,b1,w2,b2)
