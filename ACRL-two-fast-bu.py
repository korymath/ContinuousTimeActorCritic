
# coding: utf-8

# In[6]:

# get_ipython().magic(u'matplotlib inline')


# In[7]:

# import matplotlib.pyplot as plt
import numpy as np
import time
import math
from collections import Counter
from tqdm import tqdm
import gc
import sys
import collections
from collections import deque
# from joblib import Parallel, delayed
import multiprocessing as mp

import tiles

# num_cores = mp.cpu_count()
# pool = mp.Pool(processes=num_cores)
# print("numCores = " + str(num_cores))

# In[8]:

# plt.xkcd()  # Yes...

tmax = 10000

phase1maxT = 300000
phase2maxT = phase1maxT + 300000

t = tmax*np.linspace(0, 1, tmax, endpoint=False)

# Define the target angles
targetAngles = np.zeros((tmax,2))
for i in range(0, tmax, 4000): 
    targetAngles[i:i+1000,0] = 0.5
    targetAngles[i+1000:i+2000,0] = 1
    targetAngles[i+2000:i+3000,0] = 1.5
    targetAngles[i+3000:i+4000,0] = 1
    targetAngles[i:i+1000,1] = -3
    targetAngles[i+1000:i+2000,1] = -1.5
    targetAngles[i+2000:i+3000,1] = 0
    targetAngles[i+3000:i+4000,1] = -1.5
    
angleMins = np.amin(targetAngles, 0)
angleMaxs = np.amax(targetAngles, 0)

# Two subplots, unpack the axes array immediately
# plt.figure(figsize=(40,40))
# f, ax1 = plt.subplots(1, 1, sharey=False)
# ax1.set_ylim(-4,2)
# ax1.plot(t, targetAngles)
# ax1.set_title('Target Joint Signals')

# In[ ]:

delay = 0

# Noise is defined by a random normal variable 
noise = np.random.normal(0,0.001,(tmax,2))
simEMG = np.zeros((tmax+delay,2))

for i in range(1,tmax):
    if targetAngles[i,0] == 1:
        simEMG[i + delay,0] = -1
        simEMG[i + delay,1] = 1
    elif targetAngles[i,0] < 1:
        simEMG[i + delay,0] = 0
        simEMG[i + delay,1] = 0        
    elif targetAngles[i,0] > 1:
        simEMG[i + delay,0] = 1
        simEMG[i + delay,1] = -1
        
simEMGdiff = simEMG[0:tmax, :] # + noise

emgMins = np.amin(simEMGdiff, 0)
emgMaxs = np.amax(simEMGdiff, 0)

# plt.figure(figsize=(40,40))
# f, ax2 = plt.subplots(1, 1, sharey=False)
# ax2.plot(t, simEMGdiff)
# ax2.set_ylim(-1.5, 1.5)
# ax2.set_title('Simulated Noisy EMG Signal')

# In[ ]:

# Implement a learning algorithm to try to fit the signal
numJoints = 1
numEMG = 1

# define the state maximums and minimums
sMins = np.r_[[0]*numJoints, emgMins[:numEMG]]
sMaxs = np.r_[[math.pi]*numJoints, emgMaxs[:numEMG]]

jointAngle = np.zeros((tmax, numJoints))

# Initialize the state array for the trajectory
# Possible feedback from the simulated arm includes the angle
# (in radians) and angular velocity (in radians per second) of
# each joint, and the Cartesian position of the end effector.

# Initialize the continuous state space 
# composed of joint angles (thetaW,thetaE)
# and differential EMG signals (dS1,dS2)
# thetaW, thetaE, dS1, dS2
s = np.zeros(numJoints+numEMG)

# Grab the initial values of the state space
# TODO: make sure this is valuable
s = np.hstack((targetAngles[0, :numJoints], simEMG[0, :numEMG]))

# Standard deviation should cover the possible action set
maxAngVelInt_stdC = 1023

# Initialize the saved variables
reward = np.zeros(tmax)
delta = np.zeros(tmax)

agentMean = np.zeros((tmax, numJoints))
agentStd = np.zeros((tmax, numJoints))

# define the number of tilings
numTilings = 25  # 5 25

# this defines the resolution of the tiling
resolutions = np.array([5, 8, 12, 20])  # resolutions = np.array([5,8,12,20])

numFeatures = len(s)

# Initialize the learning parameters
# m is the number of active features in the feature vector
m = numTilings * len(resolutions) + 1

gamma = 0.99  # 0.97
lambd = 0.3

# Different values for the 2013 paper
lambdw = 0.3
lambdv = lambdw # 0.7

alphaV = 0.1/m
alphaW = 0.01/m

alphaU = alphaW # 0.005/m # 0.01/m
alphaS = alphaW # 0.25*alphaU # 0.01/m #

# Actual length is the concatenation at different resolutions
# and the baseline feature
featVecLength = sum(np.power(resolutions, numFeatures)*numTilings)+1

# Initialize an empty feature vector
featvec = np.zeros(featVecLength)

# Initialize a ones feature vector
fvOnes = np.ones(featVecLength)

# Initialize the weight vectors to zero 
# the should be as long as the feature vector x
elV = np.zeros(featVecLength)
v = np.zeros(featVecLength)

elU = np.zeros((featVecLength, numJoints))
elS = np.zeros((featVecLength, numJoints))
wU = np.zeros((featVecLength, numJoints))
wS = np.zeros((featVecLength, numJoints))

# This function converts the angular velocity from an integer in the range [-1023,1023]
# to an angular velocity in radians per second.
# The no-load speed of the motor is ~60RPM, which is around 6 radians per second
# rad/s = rot(RPM) 2*pi/60
# This should give a good simulation of the true robotic kinematics
def conAngIntToangVel(angInt):
    motorVel = 60
    angVel = motorVel * angInt / maxAngVelInt_stdC
    return angVel


def normalize(state):
    # Normalize the state value given the maximum and minimum possible values
    # All components in s were normalized to the range [0, 1]
    # according to their minimum and maximum POSSIBLE VALUES!

    # print state
    normS = (state - sMins) / (sMaxs - sMins)
    # print state, normS, sMins, sMaxs
    # if (any(i < 0 for i in normS)):
    #     sys.exit()

    return (state - sMins) / (sMaxs - sMins)


def idx(x, offset, res):
    # given a value and the tile offset, return the axis index from [0,res]
    return math.floor(res-1 * (x + offset))

# To capture different levels
# of generalization, x(s) was a concatenation of NT = 25
# incrementally offset tilings of s, each at four different resolution
# levels NR = [5, 8, 12, 20], along with a single active
# baseline unit

def tilecode(numTilings, res, activetiles, normS):
    for n in range(numTilings):
        offset = n * (1.0 / res) / numTilings
        idx0 = idx(normS[0], offset, res)
        idx1 = idx(normS[1], offset, res)
        # print n, normS, res, idx0, idx1
        activetiles[n] = int((res**2 * n) + (res * idx0) + idx1)

    return activetiles


def getfeatvec(res, normS):

    # activetiles is the active tiles in this featurization
    # This featurization should return a single binary feature vector
    # with exactly m features in x(s) active at any time
    # Binary feature vector has a length of 4636426 or
    # sum(np.power(np.array([5,8,12,20]),4)*25)+1

    # activetiles = np.zeros(np.power(res, numFeatures)*numTilings)

    activetiles = [-1]*numTilings
    tilesOut = tilecode(numTilings, res, activetiles, normS)

    # scalednormS = [x * res for x in normS]
    # tilesOut = tiles.tiles(numTilings, np.power(res, numFeatures)*numTilings, scalednormS)

    # for idx, val in enumerate(tilesOut):
    #     # for each tile index put out by the tile coder
    #     # we need to flip a bit in the feature vector
    #     # this bit index is given by the value of the tile from the tile coder
    #     # the index this value was at, the resolution of the tile
    #     activetiles[val + (res**2 * idx)] = 1

    return tilesOut


def featurize(s, fv):

    normS = normalize(s)

    # print len(fv)

    fv[:] = 0
    fv[0] = 1

    for res in range(len(resolutions)):
        featIdx = getfeatvec(resolutions[res], normS)
        # print featIdx

        # offset by one for the active baseline feature
        featIdx = [x+1 for x in featIdx]

        # print len(fv)

        # print sum(fv)

        fv[featIdx] = 1

        # print sum(fv)

        # test for duplicates
        # print [item for item, count in collections.Counter(featIdx).items() if count > 1]

    # # zero out the feature vector
    # fv[:] = 0
    #
    # fv[0] = 1
    #
    # startIdx = 1
    # for res in range(len(resolutions)):
    #     endIdx = startIdx + np.power(resolutions[res], numFeatures)*numTilings
    #     fv[startIdx:endIdx] = getfeatvec(resolutions[res], normS)
    #     startIdx = endIdx

    return fv


def getReward(newAngles,timeStep):
    # Define the reward function of the system
    # A positive reward of rt = 1.0 was
    # delivered when θw and θe were both within 0.1 radians of
    # their target angles. A reward of rt = −0.5 was delivered
    # in all other cases, in essence penalizing the learning system
    # when the arm’s posture differed from the target posture.

    target = targetAngles[timeStep,range(numJoints)]    
    absAngleError = np.abs(newAngles-target)
    
    if all(absAngleError < 0.1):
        r = 0.1
    else:
        r = -0.5

    return r


def perform(vel, s, timeStep):
    # take the action and observe the new state and the reward 
    # new state is defined by the new joint angle
    # which is defined by the old joint angle and the new angular velocity
    # which is applied for that time step and the
    # emg signal at that time index 
    
    # Calculate the new angular state of the joint
    # old angle + angular velocity * time (in this case time = 5ms, the period of action selection)
    # this limits the amount of motion of the joint possible in each action selection

    # Define the new state space with the new angle and the emg information from the next step
    newAngles = np.clip((s[:numJoints] + (vel[:] * 0.005)), 0, math.pi)
    # newAngles = s[:numJoints] + (vel[:] * 0.005)

    newS = np.hstack((newAngles, simEMG[timeStep, :numEMG]))
    
    # Get the reward for the new angle
    r = getReward(newAngles, timeStep)
    
    return r, newS

# process the training samples that are given
for i in tqdm(range(tmax)):
    
    jointAngle[i, :] = s[:numJoints]

    # Featurize the state
    featvec = featurize(s, featvec)

    a = np.zeros(numJoints)
    angVels = np.zeros(numJoints)

    # Calculate the mean and standard deviation of the action selection
    for j in range(numJoints):
        agentMean[i, j] = np.dot(wU[:, j], featvec)
        agentStd[i, j] = max(1, np.exp(np.dot(wS[:, j], featvec) + np.log(maxAngVelInt_stdC)))

        # get the action from the normal distribution
        # angular velocity commands are sent to joints (simulated servos)
        # as integers in the range [−maxAngVelInt_stdC, maxAngVelInt_stdC]
        a[j] = round(np.random.normal(agentMean[i, j], agentStd[i, j]))

    a = np.clip(a, -maxAngVelInt_stdC, maxAngVelInt_stdC)

    # Convert the action to an angular velocity
    angVels = conAngIntToangVel(a)

    # Take action a and observe the reward, r, and the new state, s
    reward[i], s = perform(angVels, s, i)

    if numJoints > 1:
        # Initially, parameter vectors for the wrist and elbow joints
        # were each trained in isolation for 100k time steps;
        if i < phase1maxT:
            # set joint 2 to desired position
            s[1] = targetAngles[i, 1]

        elif i < phase2maxT:
            # set joint 1 to desired position
            s[0] = targetAngles[i, 0]

    # save the old state feature vector
    x = featvec

    # update the feature vector
    featvec = featurize(s, featvec)

    # Calculate the TD Error based on the old state and the new state
    delta[i] = reward[i] + (gamma * np.dot(v, featvec)) - np.dot(v, x)

    # Critic's eligibility traces
    # Updated eligibility trace from the 2013 paper
    # replacing eligibility traces in the critic used to accelerate learning
    elV = (lambdv * elV) + x
    elV = np.minimum(fvOnes, elV)

    # Critic's parameter vector
    v += (alphaV * delta[i] * elV)

    if numJoints > 1:
    # Initially, parameter vectors for the wrist and elbow joints
    # were each trained in isolation for 100k time steps;
        if i < phase1maxT:
            # learn parameter vectors for joint 1
            # set joint 2 to desired position
            elU[:,0] = lambdw * elU[:,0] + np.multiply((a[0] - agentMean[i,0]),x)
            wU[:,0] = wU[:, 0] + alphaU * delta[i] * elU[:,0]
            elS[:,0] = lambdw * elS[:,0] + np.multiply(((np.power((a[0] - agentMean[i,0]),2) / np.power(agentStd[i,0],2)) - 1),x)
            wS[:,0] = wS[:, 0] + alphaS * delta[i] * elS[:,0]

        elif i < phase2maxT:
            # learn parameter vectors for joint 2
            # set joint 1 to desired position
            elU[:,1] = lambdw * elU[:,1] + np.multiply((a[1] - agentMean[i,1]),x)
            wU[:,1] = wU[:,1] + alphaU * delta[i] * elU[:,1]
            elS[:,1] = lambdw * elS[:,1] + np.multiply(((np.power((a[1] - agentMean[i,1]),2) / np.power(agentStd[i,1],2)) - 1),x)
            wS[:,1] = wS[:,1] + alphaS * delta[i] * elS[:,1]

        else:
            for j in range(numJoints):
                # Actors parameters (eligibiliy traces and weight vectors)
                elU[:, j] = lambdw * elU[:, j] + np.multiply((a[j] - agentMean[i, j]), x)
                wU[:, j] = wU[:, j] + alphaU * delta[i] * elU[:, j]
                elS[:, j] = lambdw * elS[:, j] + np.multiply(((np.power((a[j] - agentMean[i, j]), 2) / np.power(agentStd[i, j], 2)) - 1), x)
                wS[:, j] = wS[:, j] + alphaS * delta[i] * elS[:, j]
    else:
        for j in range(numJoints):
            # Actors parameters (eligibiliy traces and weight vectors)
            elU[:, j] = lambdw * elU[:, j] + np.multiply((a[j] - agentMean[i, j]), x)
            wU[:, j] += alphaU * delta[i] * elU[:, j]
            elS[:, j] = lambdw * elS[:, j] + np.multiply(((np.power((a[j] - agentMean[i, j]), 2) / np.power(agentStd[i, j], 2)) - 1), x)
            wS[:, j] += alphaS * delta[i] * elS[:, j]

    if (i%2000 == 0):
        print 'Step: ' + str(i)
        print np.sum(reward)
        
        
#     print 'Joint Angle: ' + str(jointAngle[i,:]) + ' rads'
#     print 'Target Angle: ' + str(targetAngles[i,range(numJoints)]) + ' rads'
#     print 'Agent Mean: ' + str(agentMean[i,:])
#     print 'Agent Std: ' + str(agentStd[i,:])
#     print 'Action: ' + str(angVels) + ' rad/s, move ' + str(angVels*0.005) + ' rads'
#     print 'New Joint Angles: ' + str(s[range(numJoints)]) + ' rads'
#     print 'Reward: ' + str(reward[i])
#     print 'TD Error: ' + str(delta[i])
#     print '\n'


# In[ ]:

# Visualize the learning    
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t, targetAngles[:,range(numJoints)])
# ax1.set_title('Target Angles')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t, jointAngle)
# ax1.set_title('Learned Joint Angles')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t, delta)
# ax1.set_title('TD Error')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t, np.cumsum(reward))
# ax1.set_title('Cumulative Return')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t, agentMean)
# ax1.set_title('Agent Mean')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t, agentStd)
# ax1.set_title('Agent Standard Deviation')


# ##### 

# In[ ]:

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t[-30000:], jointAngle[-30000:])
# ax1.plot(t[-30000:], targetAngles[-30000:])
# ax1.set_title('Target and Learned Angles')
# ax1.set_ylim([0, 2])
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t[0:30000], agentMean[0:30000])
# ax1.set_title('Agent Mean')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t[0:30000], agentStd[0:30000])
# ax1.set_title('Agent Standard Deviation')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t[-30000:], agentMean[-30000:])
# ax1.set_title('Agent Mean')
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(t[-30000:], agentStd[-30000:])
# ax1.set_title('Agent Standard Deviation')

