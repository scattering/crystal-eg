"""
Filename: epGreedyAnnealing.py
Authors: Ryan Cho and Telon Yan
Implements a modified epsilon-greedy algorithm for pycrysfml, but can be
used for similar scenarios. For crystallography, since one may not visit
individual HKL values more than once, what happens is:
We have a 1 - epsilon + k*epsilon/n chance of visiting a state/machine
whose expected reward is highest (e.g. 0.5, there may be ties) and an
epsilon/n chance of visiting one of the states/slot machines whose
expected reward is not the highest (where k is the number of states with
a tied max reward, such as 0.5, and n is the total number of possible HKL
values to go to next).
Code borrowed from:
https://imaddabbura.github.io/blog/data%20science/2018/03/31/epsilon-Greedy-Algorithm.html
https://github.com/scattering/crystal-rl
https://github.com/scattering/pycrysfml
"""

import os,sys;sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import random
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#So apparently you just can't run pycrysfml on Windows because you'd have to build all of its dependencies first
import fswig_hklgen as H
import hkl_model as Mod
import sxtal_model as S

import  bumps.names  as bumps
import bumps.fitters as fitter
from bumps.formatnum import format_uncertainty_pm

#Crystal model stuff

np.seterr(divide="ignore",invalid="ignore")

#Set data files
DATAPATH = os.path.dirname(os.path.abspath(__file__))
backgFile = None
observedFile = os.path.join(DATAPATH,r"../prnio.int")
infoFile = os.path.join(DATAPATH,r"../prnio.cfl")

#Read data
spaceGroup, crystalCell, atomList = H.readInfo(infoFile)
# return wavelength, refList, sfs2, error, two-theta, and four-circle parameters
wavelength, refList, sfs2, error = S.readIntFile(observedFile, kind="int", cell=crystalCell)
tt = [H.twoTheta(H.calcS(crystalCell, ref.hkl), wavelength) for ref in refList]
backg = None
exclusions = []


def setInitParams(zInit):
    #Make a cell
    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)
    #Define a model
    m = S.Model([], [], backg, wavelength, spaceGroup, cell,
                [atomList], exclusions,
                scale=0.062978, error=[],  extinction=[0.000105])
    #Set a range on the x value of the first atom in the model
    m.atomListModel.atomModels[0].z.value = zInit #zApprox
    m.atomListModel.atomModels[0].z.range(0,0.5)
#    m.atomListModel.atomModels[0].B.range(0,5)
#    m.atomListModel.atomModels[1].B.range(0,5)
#    m.atomListModel.atomModels[2].B.range(0,5)
#    m.atomListModel.atomModels[3].B.range(0,5)
#    m.atomListModel.atomModels[4].B.range(0,5)
#    m.atomListModel.atomModels[5].B.range(0,5)
    return m

def fit(model):
    #Create a problem from the model with bumps, then fit and solve it
    problem = bumps.FitProblem(model)
    monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))
    fitted = fitter.LevenbergMarquardtFit(problem)
    x, dx = fitted.solve(monitors=[monitor])
    return x, dx, problem.chisq()

#Beacuse I don't actually know how to use pycrysfml and bumps yet and just want something working, here's a fakeFit method that fakes the fit() method
def fakeFit(model):
    #x should be the predicted hkl positon, dx is the error, and chisq is the chi squared value
    #model should be of class "Model" in the sxtal_model program, but for now it doesn't matter
    x = random.random()
    dx = random.random()
    chisq = random.random()*150
    return x, dx, chisq


class EpsilonGreedy():
    
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values
        self.visited = []
        return
    
    def reset(self):
        #self.counts = counts
        #self.values = values
        self.visited = []
        return

    def getCounts(self):
        return self.counts

    def getValues(self):
        return self.values

    #Returns the indices of the HKL values with the best immediate reward
    #Returns the indices of the HKL values with the best immediate reward
    def bestReward(self):
        #rewardMax = -99999
        maxIndices = []
        choices = list(self.values)
        choices_indices = []
        for i in range(len(self.values)):
            choices_indices.append(i)

        popTimes = 0
        for i in self.visited:
            #print(i-popTimes)
            choices_indices.remove(i)
            choices.pop(i-popTimes)
            popTimes += 1

        rewardMax = choices[np.argmax(choices)]
        #for i in range(len(choices)):
        #    if choices[i] > rewardMax:
        #        rewardMax = choices[i]
        for i in range(len(choices)):
            if choices[i] == rewardMax:
                maxIndices.append(choices_indices[i])
        return maxIndices
    #Chooses an HKL value to go to using bestReward, ignoring hkls already visited. Returns the index of the hkl chosen
    def select_action(self):
        coin = random.random()
        choice = 0
        
        #Exploit - Pick among the options (tied) with the best expected reward
        if coin > self.epsilon:
#            prin)
            choice = random.choice(self.bestReward())
            self.visited.append(choice)
            self.visited.sort()
            
        #Explore - Pick a choice at random
        else:
#            print
            choices = list(self.values)
            choices_indices = []
            for i in range(len(self.values)):
                choices_indices.append(i)
            for i in self.visited:
                choices_indices.remove(i)
                
            choice = random.choice(choices_indices)
            self.visited.append(choice)
            self.visited.sort()
            
        return int(choice)

    #Updates the counts of hkls visited and their expected reward with new data
    def update(self, chosen_action, reward):
        self.counts[chosen_action] += 1
        n = self.counts[chosen_action]
        value = self.values[chosen_action]
        self.values[chosen_action] = value * (n-1.0)/n + float(reward)/n
	
	t = np.sum(self.counts)
	self.epsilon = 1 / np.log(t + 0.0000001)
        return

#agent is the EpsilonGreedy() object, actions is a list of HKLs 
#(each element of the list is a length 3 list: [h, k, l]), num_sims is an int, horizon is an int
def test_algorithm(agent, actions, num_sims, horizon, numParameters):
    zInit = 0.3
    for simulation in range(num_sims):
	print("simulation #" + str(simulation))
        agent.reset()
        total_reward = 0

        chosen_actionList = []
	observed_intensities = []
        rewards = np.zeros(horizon)

        #|--------------------------------Bumps stuff----------------------------------------------|
        model = setInitParams(zInit)
        prevChiSq = 0

        #agent.initialize(agent.getCounts(), agent.getRewards()) #this line is kinda pointless
        file = open("epGreedyResults" + str(simulation) + ".txt", "w")
        file.write("HKL Value\t\tReward\t\t\tTotalReward\tChi Squared Value\tZ Coordinate Approximation\t\tdx")

        reward = 0

        for t in range(horizon):
            #print(agent.getValues())
            #print(agent.visited)
            #This is the index of the action/hkl to go to at this timestep
            chosen_action = agent.select_action()
	    #print(chosen_action)
            chosen_actionList.append(actions[chosen_action])

            #|-Bumps stuff-|
            #feed actions[chosen_action] into bumps to get "reward" to use in agent.update() which updates expected reward
            #Find the data for this hkl value and add it to the model

	    #because refList Objects are hard to change, make a new reflist each time with the new data
            model.refList = H.ReflectionList(chosen_actionList)
            model._set_reflections()

            model.error.append(error[chosen_action])
            model.tt = np.append(model.tt, [tt[chosen_action]])

            observed_intensities.append(sfs2[chosen_action])
            model._set_observations(observed_intensities)
            model.update()

	    chiSq = 0
	    dx = 0
	    x = 0

            if t > numParameters - 1:
                x, dx, chiSq = fit(model)
                if t > numParameters:
		    reward = -1 * abs(chiSq - prevChiSq)
		    if (prevChiSq != 0 and chiSq < prevChiSq):
			reward += 1.5 * abs(chiSq - prevChiSq)
		    rewards[t] = reward
		    total_reward += reward
		    agent.update(chosen_action, reward)
                prevChiSq = chiSq
	     	
            file.write("\n" + str(chosen_actionList[t].hkl).replace("[","").replace("]","").replace(",",""))
            file.write("\t\t\t" + str(reward) + "\t\t\t" + str(total_reward) + "\t\t" + str(chiSq) + "\t\t" + str(model.atomListModel.atomModels[0].z.value) + "\t\t" + str(dx) + "\t\t" + str(error[chosen_action]))
	x1 = sfs2
	y = model.theory()
	plt.scatter(x1,y)
	plt.savefig('sfs2s' + str(simulation) + '.png') 

	zInit = model.atomListModel.atomModels[0].z.value
        file.close()
        
    return

#def __main__():
#for i in refList:
#    for j in i:
#        print(j)


#x2 = sfs2#

#model1 = setInitParams()
#model1.refList = H.ReflectionList(refList)
#model1._set_reflections()
#model1.error = error
#model1.tt  = tt
#model1._set_observations(sfs2)

#y1 = model1.theory()

#plt.scatter(x2,y1)
#plt.savefig('sfs2stest.png') 

agent = EpsilonGreedy(1, np.zeros(len(refList)), np.ones(len(refList)))
test_algorithm(agent, refList, 500, len(refList), 1)
print("done")
