import copy
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from itertools import product


if __name__ == "__main__":



    class QFA():

        def __init__(self,featureExtractorOrder=6, coupleFeatures=True, alpha = .0005, gamma = .99):
            self.actions = [0, 1]
            self.numberOfStateFeatures = 4
            self.featureExtractorOrder = featureExtractorOrder
            self.coupleFeatures = coupleFeatures
            self.gamma= gamma
            self.alpha= alpha

            if coupleFeatures:
                t1 = np.zeros(featureExtractorOrder**self.numberOfStateFeatures,np.longdouble)
                t2 = np.zeros(featureExtractorOrder**self.numberOfStateFeatures,np.longdouble)
                # t1 = np.append(t1,1)
                # t2 = np.append(t2,1)

            else:
                t1 = [0 for x in range (featureExtractorOrder*self.numberOfStateFeatures)]
                t2 = [0 for x in range (featureExtractorOrder*self.numberOfStateFeatures)]
            #add bias terms
            # t1.append(1)
            # t2.append(1)

            self.theata = np.asarray([t1,t2],np.longdouble)
            print()

        def forierFeatureExtractor(self,state):
            features = []
            # state[0] = np.clip(state[0],-4.8,4.8)#/(4.8)
            # state[1] = np.clip(state[1],-15,15)#/15
            # state[2] = np.clip(state[2],-0.418,0.418)#/0.418
            # state[3] = np.clip(state[3],-15,15)#/15

            state = copy.deepcopy(state)
            clip2 = 1.1
            #print("org: " + str(state))
            state[0] = (np.clip(state[0], -4.8, 4.8) + 4.8) / (2 * 4.8)
            state[1] = (np.clip(state[1], -clip2, clip2) + clip2) / (2 * clip2)
            state[2] = (np.clip(state[2], -0.418, 0.418) + 0.418) / (2*0.418)
            state[3] = (np.clip(state[3], -clip2, clip2) + clip2) / (2 * clip2)

            #print("norm: "  + str(state))

            if not self.coupleFeatures:
                for i in range(self.featureExtractorOrder):
                    for s in state:
                        features.append(np.cos(np.pi*i*s))
                        #features.append(np.sin(np.pi*i*s))
                return np.asarray(features)
            else:
                basisOrderComboVect = [x for x in range(self.featureExtractorOrder)]
                couplingMatrix = np.asarray(list(product(basisOrderComboVect, repeat=self.numberOfStateFeatures)))
                result = np.cos(np.pi*couplingMatrix@np.asarray(state))+np.sin(np.pi*couplingMatrix@np.asarray(state))#np.append(np.cos(np.pi*couplingMatrix@np.asarray(state)),np.sin(np.pi*couplingMatrix@np.asarray(state)))
                #result = np.append(result,1)
                return result

        def getQVal(self,state,action):
            feats = self.forierFeatureExtractor(state)
            return self.theata[action]@feats

        def getV(self,state):
            return np.mean([self.getQVal(state,a) for a in self.actions])

        def getMaxQ(self,state):
            return max([self.getQVal(state,a) for a in self.actions])

        def getAction(self,state):
            return np.argmax(np.asarray([self.getQVal(state,a) for a in self.actions]))

        def epGreedPolicy(self,state, epsilon=.1):
            if np.random.rand() < epsilon:
                return random.randint(0, 1)
            else:
                return self.getAction(state)

        def update(self,state,action,reward,newState,newAction=None):
            if newAction is None:
                d = reward + self.gamma * self.getMaxQ(newState) - self.getQVal(state, action)
            else:
                d = reward + self.gamma * self.getQVal(newState,newAction) - self.getQVal(state, action)
            theata = self.theata.tolist()

            phi = self.forierFeatureExtractor(state)
            theata[action] = (theata[action] + (self.alpha * d * phi)).tolist()
            self.theata = np.asarray(theata, np.longdouble)



    class PG():
        def __init__(self,env, alpha = .02,gamma = .9, featureExtractorOrder=4, coupleFeatures=True):

            self.alpha = alpha
            self.gamma = gamma
            self.actions = [0, 1]
            self.numberOfStateFeatures = 4
            self.featureExtractorOrder = featureExtractorOrder
            self.coupleFeatures = coupleFeatures
            #initilize policy probability to all 0

            if coupleFeatures:
                t1 = np.zeros(featureExtractorOrder ** self.numberOfStateFeatures, np.longdouble)
                t2 = np.zeros(featureExtractorOrder ** self.numberOfStateFeatures, np.longdouble)


            else:
                t1 = [0 for x in range(featureExtractorOrder * self.numberOfStateFeatures)]
                t2 = [0 for x in range(featureExtractorOrder * self.numberOfStateFeatures)]
            # add bias terms
            self.theata = np.asarray([t1,t2])
            self.w = np.zeros(featureExtractorOrder ** self.numberOfStateFeatures, np.longdouble)

        def update(self,state,action,t,g):
            # A = q.getQVal(state,action) - q.getV(state)
            # Ap = q.getQVal(state,self.actions[action-1]) - q.getV(state)
            # qv = q.getQVal(state,action)



            actionDistro = self.theata @ self.forierFeatureExtractor(state)
            actionDistro = self.softmax(actionDistro)

            #self.theata[action] = self.theata[action] + self.gamma**t*self.alpha*g*(1-actionDistro[action])*self.forierFeatureExtractor(state)
            # #action not taken
            #self.theata[action-1] = self.theata[action-1] + self.gamma**t*self.alpha*A*(actionDistro[action-1]*self.forierFeatureExtractor(state))

            #self.theata = self.theata + self.alpha*qv*np.expand_dims((1-actionDistro),1)@np.expand_dims(self.forierFeatureExtractor(state),0)

            # if action == 1:
            #     self.theata[1] += self.alpha * self.gamma ** t * g * (1 - actionDistro[1]) *self.forierFeatureExtractor(state)
            #     self.theata[0] += self.alpha * self.gamma ** t * g * (- self.forierFeatureExtractor(state) * actionDistro[0])
            # else:
            #     self.theata[0] += self.alpha * self.gamma ** t * g * (1 - actionDistro[0])*self.forierFeatureExtractor(state)
            #     self.theata[1] += self.alpha * self.gamma ** t * g * (-self.forierFeatureExtractor(state) * actionDistro[1])


            self.theata[action] = self.theata[action] + self.alpha * self.gamma**t * g *(1-actionDistro[action])*self.forierFeatureExtractor(state)
            self.theata[action-1] = self.theata[action-1] + self.alpha * self.gamma**t * g *(-actionDistro[action-1]*self.forierFeatureExtractor(state))

            # self.theata[action] = self.theata[action]/np.linalg.norm(self.theata[action])
            # self.theata[action-1] = self.theata[action-1]/np.linalg.norm(self.theata[action-1])

            #self.theata = self.theata-self.theata*.01

        def softmax(self,row):
            t = np.sum(np.exp(row-max(row)))
            f_x = np.array([np.exp(xi- max(row)) / t for xi in row])
            return f_x



        def getAction(self,state):
            actionDistro = self.theata@self.forierFeatureExtractor(state)
            return np.random.choice(self.actions,p=self.softmax(actionDistro))

        def forierFeatureExtractor(self,state):
            features = []
            # state[0] = np.clip(state[0],-4.8,4.8)#/(4.8)
            # state[1] = np.clip(state[1],-15,15)#/15
            # state[2] = np.clip(state[2],-0.418,0.418)#/0.418
            # state[3] = np.clip(state[3],-15,15)#/15

            state = copy.deepcopy(state)
            #print("org: " + str(state))
            state[0] = (np.clip(state[0], -4.8, 4.8) + 4.8) / (2 * 4.8)
            state[1] = (np.clip(state[1], -5, 5) + 5) / (2 * 5)
            state[2] = (np.clip(state[2], -0.418, 0.418) + 0.418) / (2*0.418)
            state[3] = (np.clip(state[3], -5, 5) + 5) / (2 * 5)

            #print("norm: "  + str(state))

            if not self.coupleFeatures:
                for i in range(self.featureExtractorOrder):
                    for s in state:
                        features.append(np.cos(np.pi*i*s))
                        #features.append(np.sin(np.pi*i*s))
                return np.asarray(features)
            else:
                basisOrderComboVect = [x for x in range(self.featureExtractorOrder)]
                couplingMatrix = np.asarray(list(product(basisOrderComboVect, repeat=self.numberOfStateFeatures)))
                result = np.cos(np.pi*couplingMatrix@np.asarray(state))+np.sin(np.pi*couplingMatrix@np.asarray(state))#np.append(np.cos(np.pi*couplingMatrix@np.asarray(state)),np.sin(np.pi*couplingMatrix@np.asarray(state)))
                #result = np.append(result,1)
                return result





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
    runsR = []
    theatasR = []

    numRuns = 10

    patience = 0

    for r in range(numRuns):

        rewards = []
        theatas = []

        pg =PG(env,featureExtractorOrder= 5, coupleFeatures=True,alpha=.01,gamma=.9)
        #qfa = QFA(featureExtractorOrder=4, coupleFeatures=True,gamma=.99,alpha=.0015)

        for e in range(numberOfEpisodes):

            #qfa = QFA(featureExtractorOrder=4, coupleFeatures=True)
            state = env.reset()
            action = pg.getAction(state)
            episodeRewards = []
            traj = []

            print(e)

            for i in range(lenOfEpisodes):
                # if numberOfEpisodes - e <=10:
                #     env.render()
                #env.render()

                newState, reward, done, info = env.step(action)
                newAction = pg.getAction(newState)

                #qfa.update(state,action,reward,newState, newAction=newAction)

                state = newState
                action = newAction
                episodeRewards.append(reward)


                # if e %1000 == 0:
                #     print(str(e)+ ": " + str(np.mean(rewards)))

                traj.append((state,action))
                # theatas.append(np.linalg.norm(theata))

                if done:
                    newState = env.reset()
                    print('E: ' +str(e) + ' I: ' + str(i))
                    print(np.average(rewards[-100:]))
                    break

            rewards.append(sum(episodeRewards))

            gs = []
            for t in range(len(episodeRewards)):
                g = 0
                for k in range(t + 1, len(episodeRewards) + 1):
                    g = g + pg.gamma ** (k - t - 1) * episodeRewards[k - 1]
                gs.append(g)
                t = t + 1

            A = []
            for i in range(len(episodeRewards)):
                v = np.average(gs[i:])
                A.append(gs[i]-v)

            t = 0
            for state,action in traj:
                #qfa.update(state, action, reward, newState)
                pg.update(state,action,t,A[t])
                t = t + 1
                theatas.append(np.linalg.norm(pg.theata))

            # if (np.max(rewards[-100:]) - np.min(rewards[-100:])) < 30 and (e > patience + 100):
            #     patience = patience + 100
            #     pg.alpha = pg.alpha*np.random.choice([2,.5],p=[.5,.5])
            #     print('lr reduced on plateau')

            if e % 100 and max(rewards) < 100:
                pg = PG(env, featureExtractorOrder=5, coupleFeatures=True, alpha=.01, gamma=.9)


            if np.average(rewards[-100:]) > 195 and e > 100:
                print("Converged with avg: " + str(np.average(rewards[-100:])) + " in " + str(e) + " steps")
                break
        runsR.append(rewards)
        theatasR.append(theatas)

    env.close()

    for x in range(len(runsR)):
        plt.plot(runsR[x],label="run "+str(x))

    plt.legend()
    plt.xlabel("Episode #")
    plt.ylabel("Reward")
    plt.title("Sum of rewards Across episodes "
              "\n gamma = " + str(pg.gamma) + "alpha " + str(pg.alpha))
    plt.show()

    # #just plot last 200
    # plt.plot(rewards[-200:])
    # plt.xlabel("Episode #")
    # plt.ylabel("Reward")
    # plt.title("Sum of rewards Across LAST 200 episodes "
    #           "\n gamma = " + str(gamma) + "alpha " + str(alpha))
    # plt.show()

    for x in range(len(theatasR)):
        plt.plot(theatasR[x],label="run "+str(x))

    plt.legend()
    plt.xlabel("Episode #")
    plt.ylabel("norm of theata")
    plt.title("norm of theata Across episodes "
              "\n gamma = " + str(pg.gamma) + "alpha " + str(pg.alpha))
    plt.show()

    print(np.average(rewards[-100:]))