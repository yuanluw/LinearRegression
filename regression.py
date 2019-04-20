# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 20:31:56 2019

@author: matt
"""

from numpy import *
from sgd import *
import matplotlib.pyplot as plt
def loadDataSet(filename):
	numFeat = len(open(filename).readline().split('\t'))-1
	dataMat = [];labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return mat(dataMat),mat(labelMat)

def ridgeRegres(xMat,yMat,lam=0.2):
	xTx = xMat.T * xMat
	#加上λI 从而使得矩阵非奇异
	denom = xTx + eye(shape(xMat)[1])*lam
	theta = denom.I * (xMat.T*yMat.T)
	return theta

def standRegres(xMat,yMat):
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0.0:
		print("This Matrix is singular,cannot do inverse")
		return
	ws = xTx.I*(xMat.T*yMat.T)
	return ws


#加载数据
X,Y = loadDataSet('ex0.txt')
ws = standRegres(X,Y)
#得到预测值
yHat = X*ws
#计算预测值和真实值相关性
print(corrcoef(yHat.T,Y))
#使用岭回归计算
ridgeWs = ridgeRegres(X,Y,lam=0.2)
yRidgeHat = X*ridgeWs
print('岭回归:',corrcoef(yRidgeHat.T,Y))

#使用sgd算法训练
model = linearRegression()
lossHistory = model.trainSGD(X,Y,learningRate=1e-3,numIters=5,verbose=True)
ax = plt.subplot(211)
ax.plot(lossHistory)

sgdHat = X*model.w.T
print('sgd: ',corrcoef(sgdHat.T,Y))
print("权值 w0:%f w1:%f"%(model.w[0,0],model.w[0,1]))


#使用bgd算法训练
model.w = None
lossHistory = model.trainBGD(X,Y,learningRate=1e-4,numIters=5000,verbose=True)
ax2 = plt.subplot(212)
ax2.plot(lossHistory)

bgdHat = X*model.w
print('bgd: ',corrcoef(bgdHat.T,Y))
plt.show()