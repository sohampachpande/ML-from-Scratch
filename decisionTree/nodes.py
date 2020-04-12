# Code to implement Node class for Decision Tree implementation

class Node:
    """
    Decision Nodes which are split into two nodes
    """
    def __init__(self, featureIndex, featureSplitValue, giniImpurity): #, data, leftNode, rightNode):
        self.featureIndex = featureIndex
        self.featureSplitValue = featureSplitValue
        self.giniImpurity = giniImpurity
        self.leftNode = None
        self.rightNode = None
        
    def setChildren(self, leftNode, rightNode):
        self.leftNode = leftNode
        self.rightNode = rightNode
        
    def getGini(self):
        return self.giniImpurity

    def getNextNode(self, testData):
        if testData[self.featureIndex]<=self.featureSplitValue:
            return self.leftNode
        else:
            return self.rightNode

class LeafNode:
    def __init__(self, label):
        self.label = label

    def getLabel(self):
        return self.label
