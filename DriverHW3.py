import gym
import gym_cliffwalking
import numpy as np
import matplotlib.pyplot as plt

def getDL(state,newState,V,gamma):
    return reward + gamma*V[newState] - V[state]







if __name__ == '__main__':

    neuForLam = {}
    for lam in [0,.3,.5,.7,1]:
        env = gym.make('gym_cliffwalking:cliffwalking-v0')
        numberOfEpisodes = 1000
        lenOfEpisodes = 200

        Vu = {x:0 for x in range(env.observation_space.n)}
        ZL = [0 for x in range(env.observation_space.n)]
        gamma = .9
        stepSize = .1
        avgNeuArr = []
        NEUBatchArr = []

        for e in range(numberOfEpisodes):

            ZL = [0 for x in range(env.observation_space.n)]

            if e % 10 == 0:
                print(np.mean(NEUBatchArr))
                avgNeuArr.append(np.mean(NEUBatchArr))
                NEUBatchArr = []

            NEUArr = []
            state = env.reset()
            for i in range(lenOfEpisodes):
                env.render()
                #should randomly take 1 of the 4 actions equal to the 25% probability
                action = env.action_space.sample()  # your agent here (this takes random actions)
                newState, reward, done, info = env.step(action)

                DL = getDL(state,newState, Vu, gamma)
                ZL[state] = ZL[state] + 1

                NEUArr.append((DL * ZL[state]) ** 2)

                for s in range(env.observation_space.n):
                    Vu[s] = Vu[s] + stepSize * ZL[s] * DL
                    ZL[s] = lam*gamma*ZL[s]

                ##NEU step


                if reward == -100:
                    newState = env.reset()


                state = newState

            NEU = 1 / lenOfEpisodes * sum(NEUArr)
            NEUBatchArr.append(NEU)

        env.close()

        neuForLam[lam] = avgNeuArr

    l = []
    for k in neuForLam.keys():
        plt.plot(neuForLam[k])
        l.append("lambda: " + str(k))

    plt.legend(l)
    plt.show()