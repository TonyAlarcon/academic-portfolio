#!/usr/bin/env python
# coding: utf-8

# # Apriori Algorithm

# In[1]:


TDB = [['a','b'],
       ['b','c','d'],
       ['a','c','d','e'],
       ['a','d','e'],
       ['a','b','c'],
       ['a','b','c','d'],
       ['a'],
       ['a','b','c'],
       ['a','b','d'],
       ['b','c','e']]

min_sup = 2


# In[2]:


from collections import Counter

class Apriori:

    def getCandidateOne(self, TDB):
        temp = []
        candidateSet = set()
        for transaction in TDB: #parses all transaction int Database
            for item in transaction: #parses all item in a parituclar treansaction 
                if item not in temp: #if the item is not in the candidate one list, we append
                    temp.append(item)
                    candidateSet.add(frozenset(item))


        return candidateSet


    def getCandidateSet(self, frequentSet, length_k):
        candidateSet = set()
        frequentSetList = list(frequentSet)

        for i in range(0, len(frequentSetList)):
            for j in range(i+1, len(frequentSetList)):
                itemset = frequentSetList[i].union(frequentSetList[j])
                if len(itemset) == length_k:
                    candidateSet.add(itemset)

        return candidateSet

    def getFrequentSet(self, candidateSet, TDB, minSup):
        candidateSetList = list(candidateSet)
        candidateSetWithSupport = Counter()
        frequentSetWithSupport = Counter()

        for itemset in candidateSetList:
            for transaction in TDB:
                transactionSet = set(transaction)
                if itemset.issubset(transaction):
                    candidateSetWithSupport[itemset] += 1

        for i in candidateSetWithSupport:
            if(candidateSetWithSupport[i] >= minSup):
                frequentSetWithSupport[i] += candidateSetWithSupport[i]

        return frequentSetWithSupport
    


    def printResults(self, frequentset, k):
        print(k-1,'- itemset:')
        for i in frequentset:
            print(str(list(i)) + ":" + str(frequentset[i]))


    def compute(self, TDB, minSup):
        candidate_one = self.getCandidateOne(TDB)
        frequent_one = self.getFrequentSet(candidate_one, TDB, minSup)
    
        currentFrequent = frequent_one
        k = 2

        while(len(currentFrequent) != 0 ):
            self.printResults(currentFrequent, k)


            candidateSet = self.getCandidateSet(currentFrequent, k)
            currentFrequent = self.getFrequentSet(candidateSet, TDB, minSup)

            k += 1


# In[3]:


model = Apriori()
model.compute(TDB, 2)

