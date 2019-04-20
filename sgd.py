# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:48:08 2019

@author: matt
"""
import numpy as np
class linearRegression(object):
	def __init__(self):
		self.w = None

	#使用SGD
	def trainSGD(self,X,Y,learningRate=1e-3,numIters=500,verbose=False):
		xArray = np.array(X)
		m,n= xArray.shape
		if self.w is None:
			self.w = np.ones(n)
		lossHistory = []
		for i in range(numIters):
			dataIndex = list(range(m))
			for j in range(m):
				learningRate = 4/(1.0+j+i) + 0.01
				randIndex = int(np.random.uniform(0,len(dataIndex)))
				h = np.dot(xArray[randIndex],self.w.T)
				loss = Y[0,randIndex] - h
				lossHistory.append(loss)
				self.w = self.w + learningRate*loss*X[randIndex]
				del(dataIndex[randIndex])
			if verbose and i%100 == 0:
				print("迭代次数：%d/%d loss: %f"%(i,numIters,lossHistory[-1]))
		return lossHistory

	#使用BGD
	def trainBGD(self,X,Y,learningRate=1e-3,reg=0.0,numIters=1000,verbose=False):
		m,n = X.shape
		xArray = np.array(X)
		if self.w is None:
			self.w = np.ones((n,1))
		lossHistory = []
		for i in range(numIters):
			h = np.dot(xArray,self.w)
			loss = Y[0].T-h
			self.w = self.w + learningRate*np.dot(xArray.T,loss)
			lossHistory.append(np.sum(loss))
			if verbose and i%100==0:
				print("迭代次数:%d/%d  loss:%f"%(i,numIters,np.sum(loss)))

		return lossHistory

