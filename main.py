import numpy as np
import matplotlib.pyplot as plt
import argparse
from cycler import cycler
import scipy.stats
import fit

c = plt.get_cmap('viridis').colors[::-1][40::35]
plt.rcParams['axes.prop_cycle'] = cycler(color=c)


class Environment:
    def __init__(self, lb, ub, noption):
        # p of each option
        self.p = np.linspace(lb, ub, noption)
        #np.random.shuffle(self.p)

    def play(self, idx):
        # for convenience we only use -1 and + 1 rewards
        return np.random.choice((-1, 1), p=[1 - self.p[idx], self.p[idx]])


class Agent:

    """
    Agent abstract model
    """

    def __init__(self, *args, **kwargs):
        self.noption = kwargs['noption']
        self.tmax = kwargs['tmax']

        self.beta = kwargs['beta']
        self.value = np.zeros((kwargs['noption'], kwargs['tmax']))

        # Data to store
        self.choice = np.zeros(kwargs['tmax'], dtype=int)
        self.reward = np.zeros(kwargs['tmax'], dtype=int)
        self.regret = np.zeros(kwargs['tmax'], dtype=int)

    def make_choice(self):
        """
        returns a choice based on softmax output
        :return:
        """
        return np.random.choice(range(self.noption), p=self.softmax())

    def softmax(self):
        """
        use prior means/ qvalues in order to compute options' probabilities
        :return:
        """
        return np.exp(
            self.beta * self.value[:] + 1e-20
            ) / sum(np.exp(self.beta * self.value[:] + 1e-20))

    def learn(self, *args, **kwargs):
        raise NotImplementedError

    def remember(self, reward, choice, t):
        self.reward[t] = reward
        self.choice[t] = choice
        self.regret[t + 1] = self.regret[t] + (1 - reward)


class QLearningAgent(Agent):
    def __init__(self, alpha, q0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.q = np.ones(kwargs['noption']) * q0
        self.value = self.q

    def learn(self, idx, reward):
        """
        :param idx:
        :param reward:
        :return:
        """
        self.q[idx] = self.q[idx] + self.alpha * (reward - self.q[idx])


class KalmanAgent(Agent):
    def __init__(self, kg0, mu0, v0, sig_xi, sig_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parameters
        # Kalman gain, could be considered as a dynamic 'learning rate'
        self.kg = np.ones(kwargs['noption']) * kg0
        # mean of reward pdf
        self.mu = np.ones(kwargs['noption']) * mu0
        self.value = self.mu
        # variance
        self.v = np.ones(kwargs['noption']) * v0
        # innovation variance
        self.sig_xi = sig_xi
        # noise variance
        self.sig_eps = sig_eps
        # stochasticity
        self.beta = kwargs['beta']

    def update_pdf(self, idx, reward):
        """
        computes posterior mean and variance
        :param idx:
        :param reward:
        :return:
        """
        self.mu[idx] = self.mu[idx] + self.kg[idx] * (reward - self.mu[idx])
        self.v[idx] = (1 - self.kg[idx]) * (self.v[idx] + self.sig_xi)

    def update_kg(self, idx):
        """
        updates the Kalman gain
        :param idx:
        :return:
        """
        self.kg[idx] = (
            self.v[idx] + self.sig_xi + 1e-20) / (
                (self.v[idx] + self.sig_xi + self.sig_eps) + 1e-20)

    def learn(self, idx, reward):
        self.update_kg(idx)
        self.update_pdf(idx, reward)


def plot(regret, sem):
    # color = [f'C{i}' for i in range(noption)]

    fig = plt.figure(figsize=(15, 12))
    ax = fig.subplots()

    for agent, title in zip([0, 1], ('Kalman Filter', 'QLearning')):
        ax.plot(regret[agent, :], lw=1.5, label=title)
        ax.fill_between(
            range(len(regret[agent, :])),
            y1=regret[agent, :] - sem[agent, :],
            y2=regret[agent, :] + sem[agent, :],
            alpha=0.5
        )
        ax.legend()
        ax.spines['right'].set_visible(0)
        ax.spines['top'].set_visible(0)
        ax.set_xlabel('trials')
        ax.set_ylabel('cumulative regret')

    plt.show()


def main(force=False):
    noption = 10
    tmax = 1000
    nagent = 50
    # best = [0.99, 0.52, 52, 0.5, 52]

    sig_eps, sig_xi, beta1, alpha, beta2 = \
        fit.get_parameters(force=force)

    env = Environment(
        lb=.01, ub=0.95, noption=noption
    )

    regret = np.zeros((2, nagent, tmax))

    # Run
    for i in range(2):
        for n in range(nagent):
            if not i:
                agent = KalmanAgent(
                    noption=noption,
                    tmax=tmax,
                    kg0=0,
                    mu0=0,
                    v0=0,
                    sig_eps=sig_eps,
                    sig_xi=sig_eps,
                    beta=beta1,
                )
            else:
                agent = QLearningAgent(
                    noption=noption,
                    tmax=tmax,
                    q0=0,
                    alpha=alpha,
                    beta=beta2
                )

            for t in range(tmax):
                choice = agent.make_choice()
                reward = env.play(choice)

                # only update if there's a new turn
                if t < tmax - 1:

                    agent.learn(
                        choice,
                        reward
                    )

                    agent.remember(
                        reward=reward,
                        choice=choice,
                        t=t
                    )

            regret[i, n, :] = agent.regret[:]

    f_regret = np.zeros((2, tmax))
    sem = np.zeros((2, tmax))

    for a in range(2):
        for t in range(tmax):

            f_regret[a, t] = np.mean(regret[a, :, t])
            sem[a, t] = scipy.stats.sem(regret[a, :, t])

    # plot
    plot(regret=f_regret, sem=sem)


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", action="store_true",
                        help="force optimization")
    args = parser.parse_args()

    is_running_in_pycharm = "PYCHARM_HOSTED" in os.environ

    if is_running_in_pycharm:
        parser.print_help()

    main(force=args.force)
