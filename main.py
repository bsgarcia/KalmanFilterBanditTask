import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

c = plt.get_cmap('viridis').colors[::-1][40::35]
plt.rcParams['axes.prop_cycle'] = cycler(color=c)


class Environment:
    def __init__(self, lb, ub, noption):
        # p of each option
        self.p = np.linspace(lb, ub, noption)
        np.random.shuffle(self.p)

    def play(self, idx):
        # for convenience we only use -1 and + 1 rewards
        return np.random.choice((-1, 1), p=[1-self.p[idx], self.p[idx]])


class Agent:
    def __init__(self, noption, tmax, kg0, mu0, v0, sig_xi, sig_eps, beta):
        # Parameters
        self.noption = noption
        # Kalman gain, could be considered as a dynamic 'learning rate'
        self.kg = np.ones((noption, tmax)) * kg0
        # mean of reward pdf
        self.mu = np.ones((noption, tmax)) * mu0
        # variance
        self.v = np.ones((noption, tmax)) * v0
        # innovation variance
        self.sig_xi = sig_xi
        # noise variance
        self.sig_eps = sig_eps
        # stochasticity
        self.beta = beta
        # Data to store
        self.choice = np.zeros(tmax, dtype=int)
        self.reward = np.zeros(tmax, dtype=int)

    def softmax(self, t):
        """
        use prior means in order to compute options' probabilities
        :param t:
        :return:
        """
        return np.exp(self.beta * self.mu[:, t]) / sum(np.exp(self.beta * self.mu[:, t]))

    def make_choice(self, t):
        """
        returns a choice based on softmax output
        :param t:
        :return:
        """
        return np.random.choice(range(self.noption), p=self.softmax(t))

    def update_pdf(self, idx, t, reward):
        """
        computes posterior mean and variance
        :param idx:
        :param t:
        :param reward:
        :return:
        """
        self.mu[idx, t+1:] = self.mu[idx, t] + self.kg[idx, t+1] * (reward - self.mu[idx, t])
        self.v[idx, t+1:] = (1 - self.kg[idx, t+1]) * (self.v[idx, t] + self.sig_xi)

    def update_kg(self, idx, t):
        """
        updates the Kalman gain
        :param idx:
        :param t:
        :return:
        """
        self.kg[idx, t+1:] = (self.v[idx, t] + self.sig_xi) / (self.v[idx, t] + self.sig_xi + self.sig_eps)


def plot(noption, agent, env):

    fig = plt.figure(figsize=(15, 12))
    color = [f"C{i}" for i in range(noption)]

    ax = fig.add_subplot(211)
    for i in range(noption):
        ax.plot(agent.mu[i, :], label=f'option {i}', lw=1.5)
    ax.legend()
    ax.spines['right'].set_visible(0)
    ax.spines['top'].set_visible(0)
    ax.set_xlabel('trials')
    ax.set_ylabel('$\\mu$')

    ax = fig.add_subplot(223)
    count = np.zeros(noption)
    for i in range(noption):
        count[i] = sum(agent.choice == i)
    ax.bar(np.arange(noption), count, color=color)
    ax.spines['right'].set_visible(0)
    ax.spines['top'].set_visible(0)
    ax.set_xlabel('trials')
    ax.set_xlabel('options')
    ax.set_ylabel('N time chosen')

    ax = fig.add_subplot(224)
    ax.bar(np.arange(noption), env.p, color=color)
    ax.spines['right'].set_visible(0)
    ax.spines['top'].set_visible(0)
    ax.set_xlabel('trials')
    ax.set_xlabel('options')
    ax.set_ylabel('$P(R=1)$')

    plt.show()


def main():

    noption = 5
    tmax = 100

    env = Environment(lb=.01, ub=0.95, noption=noption)
    agent = Agent(noption=noption, tmax=tmax, kg0=0, mu0=0, v0=0,
                  sig_eps=0.5, sig_xi=0.5, beta=1)

    # Run
    for t in range(tmax):
        choice = agent.make_choice(t)
        reward = env.play(choice)

        # only update if there's a new turn
        if t < tmax - 1:
            agent.update_kg(choice, t)
            agent.update_pdf(choice, t, reward)
            agent.reward[t] = reward
            agent.choice[t] = choice

    # plot
    plot(noption=noption, env=env, agent=agent)


if __name__ == '__main__':
    main()
