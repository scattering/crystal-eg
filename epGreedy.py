"""
Filename: epGreedy.py
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
"""

import random
import numpy as np
import os

#For using pycrysfml
##import fswig_hklgen as H
##import hkl_model as Mod
##import sxtal_model as S


class EpsilonGreedy():
    def __init__(self, epsilon, counts, rewards):
        self.epsilon = epsilon      #chance of entering exploration
        self.counts  = counts       #number of times each place/hkl has been visited
        self.rewards = rewards      #expected reward for each possibility, associated by index
        return

    def initialize(self, counts, rewards):
        self.counts = counts #np.zeros(n_actions, dtype=int)
        self.rewards = rewards #np.zeros(n_actions, dtype=float)

    def getCounts(self):
        return self.counts

    def getRewards(self):
        return self.rewards

    #Returns the indices of the tied states/HKL values with the best immediate reward
    def bestRewards(self):
        rewardMax = 0
        rewardIndices = []
        for i in range(len(self.rewards)):
            if self.rewards[i] > rewardMax:
                rewardMax = self.rewards[i]
        for i in range(len(self.rewards)):
            if self.rewards[i] == rewardMax:
                rewardIndices.append(i)
        return rewardIndices

    #Chooses a state/HKL value to go to
    def select_action(self):
        coin = random.random()
        if coin > self.epsilon:
            #pick one of the best arms/choices and return its index
            return random.choice(self.bestRewards())
        else:									#TODO remove the possibility of selecting 
            #pick one of the choices randomly
            return random.randint(0,len(self.rewards))

    #Updates the 
    def update(self, chosenAction, reward):
        self.counts[chosenAction] += 1
        n = self.counts[chosenAction]
        #The predicted reward function is an average of all the previous rewards when
        #having visited that state/location
        self.rewards[chosenAction] = self.rewards[chosenAction]*(n-1.0)/n + float(reward)/n
        return

#The pycrysfml stuff
##class getFit():
##    def __init__(self, holy crap pycrysfml is hard):
def fit():
    #do pycrysfml stuff here
    return 1
def chiSq(observed, expected):
    chisq = 0
    for i in range(len(observed)):
        for j in range(len(observed[0])):
            chisq += float(observed[i][j]-expected[i][j])**2 / expected[i][j]
    return chisq

#To be deleted later and replaced with pycrysfml stuff - getFit()
class BernoulliArm:
    def __init__(self, p):
        self.p = p

    def draw(self):
        z = np.random.random()
        if z > self.p:
            return 0.0
        return 1.0

#Have the agent complete the "game" <numEpochs> number of times with actions and a limit of horizon # of moves per game
def test_algorithm(agent, actions, numEpochs, horizon):
    
    chosenActions = np.zeros((numEpochs, horizon))
    rewards = np.zeros((numEpochs, horizon))

    realcrystal = [[2,2,2], [2,2,1], [2,1,2], [1,2,2], [2,1,1], [1,2,1], [1,1,2], [1,1,1]]

    for epoch in range(numEpochs):
        totalReward = 0
        chiSqVal = 0
        #this is hard coded
        crystalApproximation = np.zeros((8,3))

        agent.initialize(agent.getCounts(), agent.getRewards())

        file = open("eGreedyResults" + str(epoch) + ".txt", "w")
        file.write("HKL Value\t\tReward\tTotalReward\tChi Squared Value\n")

        for t in range(horizon):
            
            action = agent.select_action()
            chosenActions[epoch][t] = action
            #do pycrysfml stuff to update crystalApproximation

            reward = fit()#actions[action].fit()
            chiSqVal = chiSq(crystalApproximation, realcrystal)#actions[action].chiSq()
            rewards[epoch, t] = reward
            totalReward += reward
            
            agent.update(chosenActions[epoch][t], reward)
            
            file.write(chosenActions[epoch, t].replace("[","").replace("]","").replace(",",""))
            file.write("\t\t" + str(reward) + "\t" + str(totalReward) + "\t" + str(chiSqVal) + "\n")
            
    return 


def __main__():
    f = open("results.txt", "w")
    random.seed(1)
    ##TODO what is this
    avgReward = [0.1, 0.2, 0.3, 0.8, 0.4]
    n_actions = len(avgReward)
    np.random.shuffle(avgReward)
    actions = list(map(lambda mu: BernoulliArm(mu), avgReward))
    
    #bestActionIndex = np.argmax(avgReward)
    
##    agent = EpsilonGreedy(0.1, [0,0,0,0,0,0,0,0], [.1,.1,.1,.1,.1,.1,.1,.1])
##    hkls = [[1, 1, 1], [1, 1, 3], [1, 3, 1], [3, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1], [3, 3, 3]]
##    test_algorithm(agent, hkls, 10, 8)

    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = EpsilonGreedy(epsilon, [], [])
        algo.initialize(n_actions, avgReward)
        results = test_algorithm(algo, actions, 500, 250)
        for i in range(len(results[0])):
            f.write(str(epsilon) + "\t")
            f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

__main__()
