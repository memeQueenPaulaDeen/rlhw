import random

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def QFASEMITable():
    Vstar = 0
    potentialRewards = [qfa.getQvalueEstimate(newState, a) for a in qfa.actions]
    Vstar = max(potentialRewards)
    estimateOfVstar = qfa.getQvalueEstimate(state, action)
    for SApair, oldEstOfVStar in qfa.extractor(state, action):
        qfa.weights[SApair] = qfa.weights[SApair] - (
                    alpha * (estimateOfVstar - (reward + gamma * Vstar)) * oldEstOfVStar)


if __name__ == "__main__":



    class QFAT():

        def __init__(self,numberOfParameters):
            self.numberOfParameters = numberOfParameters
            self.theata = [0 for x in range(numberOfParameters)]
            self.actions = [0,1]
            self.table = [0,0]
            self.weights = defaultdict(float)
            self.extractor = self.forierFeatureExtractor
            #self.table = [[0 for x in range(len(self.actions))] for i in range(numberOfParameters)]


        def forierFeatureExtractor(self,state,action,numberOfStateParams = 7):
            #state = tuple(np.cos(np.pi*i*state) for i in range(1,numberOfStateParams+1))
            features = []

            state[0] = state[0]
            #IDK how tf you do this if you cant scale inf but whatever
            state[1] = state[1]
            state[2] = state[2]
            state[3] = state[3]

            for i in range(numberOfStateParams):
                for j in range(len(state)):
                    SAPair = (i,j,tuple(round(s,1) for s in state), action)
                    valueEstimate = np.cos(np.pi*i*state[j])
                    features.append((SAPair,valueEstimate))
            return features


        def identityFeatureExtractor(self,state,action):
            features = []
            state = tuple(round(s,1) for s in state)
            features.append(((0,state,action),1))
            features.append(((1,state,action),state[0]))
            features.append(((2,state,action),state[2]))
            features.append(((3,state,action),state[1]))
            features.append(((4,state,action),state[3]))
            return features

        def getQvalueEstimate(self,state,action):
            Qvalue = 0
            for SApair, valueEstimate in self.extractor(state,action):
                Qvalue = Qvalue + self.weights[SApair] * valueEstimate
            return Qvalue

        def predictValuesForState(self,state):
            return state@self.weights

        def epGreedPolicy(self,state, epsilon=.1):
            if np.random.rand() < epsilon:
                return random.randint(0, 1)
            else:
                Qvals = [self.getQvalueEstimate(state,a) for a in self.actions]
                return np.argmax(Qvals)

        def getBasis(self,state,action,max=False):
            numFeat = 4
            result = []
            f = []
            for a in self.actions:


                for i in range(1,self.numberOfParameters+1):
                    r = np.cos(np.pi*np.matmul(np.transpose(np.asarray([i for x in range(numFeat)])),np.asarray(state)))
                    result.append(r)

                f.append(result)
                result = []

            if not max:
                return f[action]
            else:
                return np.max(np.matmul(f,self.theata))


    # class Q():
    #
    #     def __init__(self,env):
    #         self.table = {}
    #         self.bestAction = None
    #         for s in range(env.observation_space.n):
    #             for a in range(env.action_space.n):
    #                 self.table[(s,a)] = 0
    #
    #     def set(self,state, action, newQvalue):
    #         self.table[(state,action)] = newQvalue
    #
    #     def get(self,state, action):
    #         return self.table[(state,action)]
    #
    #     def getMaxWRTA(self,newState):
    #         result = -999999999999
    #         for k in self.table.keys():
    #             valueOfSApair =  self.table[k]
    #             if k[0] == newState and result <= valueOfSApair:
    #                 result = valueOfSApair
    #                 self.bestAction = k[1]
    #         return result


    env = gym.make('CartPole-v0')
    #action space is
        # Type: Discrete(2)
        # Num   Action
        # 0     Push cart to the left
        # 1     Push cart to the right
    #Observation Space is
    #Type: Box(4)
        # Num     Observation               Min                     Max
        # 0       Cart Position             -4.8                    4.8
        # 1       Cart Velocity             -Inf                    Inf
        # 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        # 3       Pole Angular Velocity     -Inf                    Inf



    numberOfEpisodes = 1000
    lenOfEpisodes = 200
    rewards = []


    alpha = .03#1/np.sqrt(numberOfEpisodes)
    gamma = .9
    qfa = QFAT(4)

    for e in range(numberOfEpisodes):

        state = env.reset()
        episodeRewards = []

        for i in range(lenOfEpisodes):
            #env.render()

            # should randomly take 1 of the 4 actions equal to the 25% probability
            if e == 0:
                action = env.action_space.sample()  # your agent here (this takes random actions)
            else:
                action = qfa.epGreedPolicy(state)
                #action = qfa.epGreedPolicy(state,epsilon=1-e/numberOfEpisodes)
                #action = np.argmax(qfa.table)

            newState, reward, done, info = env.step(action)

            if done and e != lenOfEpisodes - 1:
                reward = -reward  # punishment for losing

            #WTF
            #assuming that phi is actually the states? an ID mapping basically

            QFASEMITable()

            state = newState
            episodeRewards.append(reward)

            if done:
                newState = env.reset()
                break

        rewards.append(sum(episodeRewards))



    env.close()

    plt.plot(rewards)
    plt.xlabel("Episode #")
    plt.ylabel("Reward")
    plt.title("Sum of rewards Across episodes "
              "\n gamma = " + str(gamma) + "alpha " + str(alpha))
    plt.show()
