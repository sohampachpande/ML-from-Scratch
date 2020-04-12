import numpy as np
import pandas as pd
from nodes import Node, LeafNode
# Implementation of Classification Decision Tree using CART algorithm
class dTree:
    def __init__(self, data, maxDepth, minSizeLeafNode):

        self.data = data
        self.maxDepth = maxDepth
        self.minSizeLeafNode = minSizeLeafNode
        
        self.root = None
        self.noFeatures = data.shape[1]-1
        self.classLabels = data[data.shape[1]-1].unique().tolist()
    
    # Function to calculate gini index for a split comprising of two groups
    def giniIndex(self, splits):
        gini_index = 0.0
        totalSampleSize = len(splits[0])+len(splits[1])
        
        for split in splits:
            
            if len(split)==0:
                continue
                
            score_split = 1
            for label in self.classLabels:
                score_split -= (len(split[split[self.noFeatures]==label])/len(split))**2
            gini_index += score_split*(len(split)*1.0/totalSampleSize)
        
        return gini_index
        
    # Partition the data into 2 groups
    # split1 is all rows where row[featureIndex] less than or equal to featureSplit
    def getPartition(self, data, featureIndex, featureSplit):
        return data[data[featureIndex]<=featureSplit], data[data[featureIndex]>featureSplit]
    
    # Iterate through all possible splits of data and find the optimum split
    def getSplit(self, data):
        # define paramters to store optimum values
        optFeatureIndex, optSplitValue, optSplitGroup = None, None, None
        optGiniIndex = float('inf')
        
        for featureIndex in range(self.noFeatures):
            for row in data.iterrows():
                splitValue = row[1][featureIndex]
                split1, split2 = self.getPartition(data, featureIndex, splitValue)
                                
                gini_ = self.giniIndex([split1, split2])
                
                if gini_<optGiniIndex:
                    optFeatureIndex, optSplitValue, optSplitGroup = featureIndex, splitValue, [split1, split2]
                    optGiniIndex = gini_
                    
        return optGiniIndex, optFeatureIndex, optSplitValue, optSplitGroup
    
    def splitProcess(self, node, depth, split):
        
        nodeGini, nodeFeatureIndex, nodeSplitValue, nodeSplitGroup = None,None,None,None
        
        if ((len(split[0])==0) or (len(split[1])==0)):
            mergedSplit = pd.concat([split[0], split[1]], ignore_index=True)
            leftChild = rightChild = LeafNode(mergedSplit.mode(axis=0, dropna=True)[self.noFeatures].tolist()[0])
            node.setChildren(leftChild, rightChild)
            return
        
        # Stop if max depth is reached
        if depth==self.maxDepth:
            leftChild = LeafNode(split[0].mode(axis=0, dropna=True)[self.noFeatures].tolist()[0])
            rightChild = LeafNode(split[1].mode(axis=0, dropna=True)[self.noFeatures].tolist()[0])
            node.setChildren(leftChild, rightChild)
            return
        
        # build the left subtree
        if len(split[0])<self.minSizeLeafNode:
            leftChild = LeafNode(split[0].mode(axis=0, dropna=True)[self.noFeatures].tolist()[0])
        else:
            nodeGini, nodeFeatureIndex, nodeSplitValue, nodeSplitGroup = self.getSplit(split[0])
            leftChild = Node(nodeFeatureIndex, nodeSplitValue, nodeGini)
            self.splitProcess(leftChild, depth+1, nodeSplitGroup)
            
        # build the right subtree
        if len(split[1])<self.minSizeLeafNode:
            rightChild = LeafNode(split[1].mode(axis=0, dropna=True)[self.noFeatures].tolist()[0])
        else:
            nodeGini, nodeFeatureIndex, nodeSplitValue, nodeSplitGroup = self.getSplit(split[1])
            rightChild = Node(nodeFeatureIndex, nodeSplitValue, nodeGini)
            self.splitProcess(rightChild, depth+1, nodeSplitGroup)
        
        node.setChildren(leftChild, rightChild)
    
    def buildTree(self):
        rootGini, rootFeatureIndex, rootSplitValue, rootSplitGroup = self.getSplit(self.data);
        self.root = Node(rootFeatureIndex, rootSplitValue, rootGini)
        self.splitProcess(self.root, 1, rootSplitGroup)
        
    def predict(self, testData):
        result = []
        for row in testData.iterrows():
            temp = self.root
            while(type(temp)==Node):
                temp = temp.getNextNode(row[1])
            result.append(temp.getLabel())
        return pd.DataFrame(result)
    
    def predictAccuracy(self, testData, testDataTruth):
        result = []
        for row in testData.iterrows():
            temp = self.root
            while(type(temp)==Node):
                temp = temp.getNextNode(row[1])
            result.append(temp.getLabel())
        prediction = pd.DataFrame(result)
        
        accuracyCounts = np.unique((prediction == testDataTruth)[0].tolist(), return_counts=True)
        
        countTrue = 0
        countTotal = 0
        for i,c in zip(accuracyCounts[0].tolist(), accuracyCounts[1].tolist()):
            if i==True:
                countTrue += c
            countTotal += c
        
        return countTrue/countTotal