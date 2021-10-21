import gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":


    class QFA():

        def __init__(self,numberOfParameters):
            self.numberOfParameters = numberOfParameters
            self.theata = [0 for x in range(numberOfParameters)]
            self.actions = [0,1]
            self.table = [0,0]
            #self.table = [[0 for x in range(len(self.actions))] for i in range(numberOfParameters)]

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


    alpha = .1
    gamma = .9
    qfa = QFA(5)

    numberOfEpisodes = 400
    lenOfEpisodes = 2000
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
                action = np.argmax(qfa.table)

            newState, reward, done, info = env.step(action)

            #WTF
            d = reward + gamma*qfa.getBasis(newState,action,max=True) \
                - np.matmul(np.transpose(qfa.getBasis(state,action)),qfa.theata)

            qfa.theata = qfa.theata - alpha*d*np.asarray(qfa.getBasis(state,action))

            qfa.table[action] = np.matmul(qfa.theata,np.transpose(qfa.getBasis(state,action)))


            print(state)
            state = newState
            episodeRewards.append(reward)

            if done:
                newState = env.reset()
                break

        rewards.append(sum(episodeRewards))



    env.close()

    plt.plot(rewards)
    plt.show()
