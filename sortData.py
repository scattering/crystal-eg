"""
Sort the rows of an epGreedy output file to the right HKL order and plotting calculated sfs2 vs observed sfs2
This is hardcoded
"""
import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
import matplotlib as mpl
import matplotlit.pyplot as plt

import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S


observedFile = os.path.join(DATAPATH,r"../prnio.int")
infoFile = os.path.join(DATAPATH,r"../prnio.cfl")

#Read data
spaceGroup, crystalCell, atomList = H.readInfo(infoFile)
# return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)



#Get the ordered list of hkls
hkls = ["" for i in range(len(refList))]
for i in range(len(refList)):
    hkls[i] = str(refList[i].hkl).replace("[","").replace("]","").replace(",","")

#Read data from data file

file = open("epGreedyResults499.txt", "r")
#Gets rid of the first line of the file
file.readline()

data = file.read()
data = data.split("\n")
for i in range(len(data)):
    data[i] = data[i].split("\t")
    data[i] = filter(None, data[i])
#print(data)

newData = list(data)
