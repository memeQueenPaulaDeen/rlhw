import gym
import gym_cliffwalking
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    class Q():

        def __init__(self,env):
            self.table = {}
            self.bestAction = None
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


    env = gym.make('gym_cliffwalking:cliffwalking-v0')
    alpha = .1
    gamma = 1
    q = Q(env)

    numberOfEpisodes = 1000
    lenOfEpisodes = 50
    rewards = []

    for e in range(numberOfEpisodes):

        state = env.reset()
        episodeRewards = []

        for i in range(lenOfEpisodes):
            env.render()

            # should randomly take 1 of the 4 actions equal to the 25% probability
            if e == 0:
                action = env.action_space.sample()  # your agent here (this takes random actions)
            else:
                action = q.bestAction

            newState, reward, done, info = env.step(action)

            newQ = q.get(state,action) + alpha* (reward + gamma*q.getMaxWRTA(newState) -q.get(state,action))
            q.set(state,action,newQ)

            print(state)
            state = newState
            episodeRewards.append(reward)

            if reward == -100 or done:
                newState = env.reset()
                break

        rewards.append(sum(episodeRewards))



    env.close()

    plt.plot(rewards)
    plt.xlabel('Episode #')
    plt.ylabel('Reward for episode')
    plt.title("Q learning in a tabular setting ")
    plt.show()
