"""
This is to test for any inconsistencies in Bumps
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

def setInitParams():
    #Make a cell
    cell = Mod.makeCell(crystalCell, spaceGroup.xtalSystem)
    #Define a model
    m = S.Model([], [], backg, wavelength, spaceGroup, cell,
                [atomList], exclusions,
                scale=0.062978, error=[],  extinction=[0.000105])
    #Set a range on the x value of the first atom in the model
    m.atomListModel.atomModels[0].z.value = 0.3 #zApprox
    m.atomListModel.atomModels[0].z.range(0,0.5)
    m.atomListModel.atomModels[0].B.range(0,5)
    m.atomListModel.atomModels[1].B.range(0,5)
    m.atomListModel.atomModels[2].B.range(0,5)
    m.atomListModel.atomModels[3].B.range(0,5)
    m.atomListModel.atomModels[4].B.range(0,5)
    m.atomListModel.atomModels[5].B.range(0,5)
    return m

def fit(model):
    #Create a problem from the model with bumps, then fit and solve it
    problem = bumps.FitProblem(model)
    monitor = fitter.StepMonitor(problem, open("sxtalFitMonitor.txt","w"))
    fitted = fitter.LevenbergMarquardtFit(problem)
    x, dx = fitted.solve(monitors=[monitor])
    return x, dx, problem.chisq()

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
        choice = len(self.visited)
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

def test_algorithm(agent, actions, num_sims, horizon, numParameters):
#    zInit = 0.3
    for simulation in range(num_sims):
	print("simulation #" + str(simulation))
        agent.reset()
        total_reward = 0

	#Action list (actual ReflectionList Objects)
        chosen_actionList = []
	#Action index list
	actionIndexList = []
	observed_intensities = []
        rewards = np.zeros(horizon)
	chiSqs = []

        model = setInitParams()
        prevChiSq = 0

        #agent.initialize(agent.getCounts(), agent.getRewards()) #this line is kinda pointless
        file = open("epGreedyResults" + str(simulation) + ".txt", "w")
        file.write("HKL Value\t\tReward\t\t\tTotalReward\tChi Squared Value\tZ Coordinate Approximation\t\tdx")

        reward = 0
	qSquared = []

        for t in range(horizon):
            #print(agent.getValues())
            #print(agent.visited)
            #This is the index of the action/hkl to go to at this timestep
            chosen_action = agent.select_action()
	    #print(chosen_action)
	    actionIndexList.append(chosen_action)
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

	    chiSqs.append(chiSq)

	    h = chosen_actionList[t].hkl[0]
	    k = chosen_actionList[t].hkl[1]
	    l = chosen_actionList[t].hkl[2]
	    A = 5.417799
	    B = 5.414600
	    C = 12.483399

	    qsq = (h/A)**2 + (k/B)**2 + (l/C)**2
	    qSquared.append(qsq)

            file.write("\n" + str(chosen_actionList[t].hkl).replace("[","").replace("]","").replace(",",""))
            file.write("\t\t\t" + str(reward) + "\t\t\t" + str(total_reward) + "\t\t" + str(chiSq) + "\t\t" + str(model.atomListModel.atomModels[0].z.value))

	    if (((t > 10) and chiSq < 1.5) or (t > 100)):
		break

#	if (simulation % 10 == 0):
#	    file2 = open("Rewards" + str(simulation) + ".txt", "w")
#	    file2.write("Number of epochs: " + str(simulation))
#	    np.savetxt("Rewards" + str(simulation) + ".txt", agent.values)
#	    file2.close()

	#Observed sfs2 values (
	x1 = sfs2[0:horizon]
	y = model.theory() #H.calcstructfact()
	print(y)
	print(x1)
	N = 10
#	data = np.random.random((N, 4))
	labels = [str(round(chiSqs[i],1)) for i in range(len(chiSqs))]

#	plt.subplots_adjust(bottom = 0.1)
#	plt.scatter(
#	    qSquared, y, marker='o', cmap=plt.get_cmap('Spectral'))

#	for label, q, sfs in zip(labels, qSquared, y):#
#	    plt.annotate(
#        	label,
#	        xy=(q, sfs), xytext=(0, 20),
#	        textcoords='offset points', ha='right', va='bottom',
#	        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#	        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

	fig,ax = plt.subplots()
	ax.scatter(qSquared, y)
	
	for i, txt in enumerate(labels):
		ax.annotate(txt, (qSquared[i], y[i]), xytext = (qSquared[i], y[i]))


#	plt.scatter(qSquared,y)
	plt.savefig("Calc sfs2 vs Qsq " + str(simulation) + ".png") 
	plt.scatter(qSquared,x1)
	plt.savefig("Obs sfs2 vs Qsq " + str(simulation) + ".png")

	

#	zInit = model.atomListModel.atomModels[0].z.value
        file.close()
        
    return

agent = EpsilonGreedy(1, np.zeros(len(refList)), np.ones(len(refList)))
#test_algorithm(agent, refList, 1, len(refList), 1)
test_algorithm(agent, refList, 1, 15, 7)
print("done")
