import copy

import gym
import gym_cliffwalking
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    class Q():

        def __init__(self,env, alpha = .15,gamma = 1):
            self.table = {}
            self.bestAction = None
            self.alpha = alpha
            self.gamma = gamma
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    self.table[(s,a)] = 0

        def set(self,state, action, newQvalue):
            self.table[(state,action)] = newQvalue

        def get(self,state, action):
            return self.table[(state,action)]

        def getMaxWRTA(self,newState):
            result = -999999999999
            for k in self.table.keys():
                valueOfSApair =  self.table[k]
                if k[0] == newState and result <= valueOfSApair:
                    result = valueOfSApair
                    self.bestAction = k[1]
            return result

        def updateQ(self,state,action,reward,newState):
            newQ = self.get(state, action) + self.alpha * (reward + self.gamma * self.getMaxWRTA(newState) - self.get(state, action))
            self.set(state, action, newQ)

    class PG():
        def __init__(self,env, alpha = .25,gamma = .75):
            self.table = {}
            self.alpha = alpha
            self.gamma = gamma
            #initilize policy probability to all 0
            for s in range(env.observation_space.n):
                self.table[s] = [0 for a in range(env.action_space.n)]


        def update(self,q,state,action):
            self.table[state][action] = self.table[state][action] + self.alpha/(1-self.gamma)*q.get(state,action)\
                *(1-self.getParamitrized(state,action,self.table))

        def softmax(self,row):
            t = np.sum(np.exp(row))
            f_x = np.array([np.exp(xi) / t for xi in row])
            return f_x

        def getParamitrized(self,state,action,table):
            return self.softmax(table[state])[action]

        def getAction(self,state):
            actions = np.array([0,1,2,3])
            return np.random.choice(actions,p=self.softmax(self.table[state]))


    env = gym.make('gym_cliffwalking:cliffwalking-v0')


    pg = PG(env)

    numberOfEpisodes = 1000
    lenOfEpisodes = 200
    rewards = []

    for e in range(numberOfEpisodes):

        q = Q(env)
        state = env.reset()
        episodeRewards = []
        replay = []

        for i in range(lenOfEpisodes):
            env.render()

            # should randomly take 1 of the 4 actions equal to the 25% probability

            action = pg.getAction(state)
            newState, reward, done, info = env.step(action)

            q.updateQ(state,action,reward,newState)
            replay.append((state,action,reward,newState))

            #print(state)
            state = newState
            episodeRewards.append(reward)

            if reward == -100 or done:
                newState = env.reset()
                break

        for state, action, reward, newState in replay:

            pg.update(q,state,action)


        rewards.append(sum(episodeRewards))



    env.close()
    print([pg.softmax(pg.table[x]) for x in pg.table])

    plt.plot(rewards)
    plt.xlabel('Episode #')
    plt.ylabel('Reward for episode')
    plt.title("Policy gradient monte carlo in a tabular setting \nlr: "+str(pg.alpha) +"discount: " + str(pg.gamma))
    plt.show()
