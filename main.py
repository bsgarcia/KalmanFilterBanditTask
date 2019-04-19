import numpy as np
import matplotlib.pyplot as plt
import argparse
from cycler import cycler
import scipy.stats
import itertools as it
import fit

c = plt.get_cmap('viridis').colors[::-1][40::35]
plt.rcParams['axes.prop_cycle'] = cycler(color=c)


class Environment:
    def __init__(self, lb, ub, noption, tmax):
        # p of each option
        self.p = np.linspace(lb, ub, noption)
        np.random.shuffle(self.p)
        self.pair = list(it.permutations(range(noption), 2))
        self.ncontext = len(self.pair)
        self.con = np.repeat(range(len(self.pair)), tmax)

    def play(self, s, a):
        # for convenience we only use -1 and + 1 rewards
        return np.random.choice(
            (-1, 1),
            p=[1 - self.p[self.pair[s][a]], self.p[self.pair[s][a]]]
        )


class Agent:

    """
    Agent abstract model
    """

    def __init__(self, *args, **kwargs):
        self.noption = 2
        self.tmax = kwargs['tmax']

        self.beta = kwargs['beta']
        self.value = np.zeros((kwargs['ncontext'], self.noption))

        # Data to store
        self.reward = np.zeros(kwargs['tmax'], dtype=int)
        self.regret = np.zeros(kwargs['tmax'], dtype=int)

    def make_choice(self, s):
        """
        returns a choice based on softmax output
        :return:
        """
        return np.random.choice(range(self.noption), p=self.softmax(s))

    def softmax(self, s):
        """
        use prior means/ qvalues in order to compute options' probabilities
        :return:
        """
        return np.exp(
            self.beta * self.value[s, :] + 1e-20
            ) / sum(np.exp(self.beta * self.value[s, :] + 1e-20))

    def learn(self, *args, **kwargs):
        raise NotImplementedError

    def remember(self, r, t):
        self.reward[t] = r
        self.regret[t + 1] = self.regret[t] + (1 - r)


class QLearningAgent(Agent):
    def __init__(self, alpha, q0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.q = np.ones((kwargs['ncontext'], self.noption)) * q0
        self.value = self.q

    def learn(self, s, a, r):
        """
        :param s:
        :param a:
        :param r:
        :return:
        """
        self.q[s, a] += self.alpha * (r - self.q[s, a])


class KalmanAgent(Agent):
    def __init__(self, kg0, mu0, v0, sig_xi, sig_eps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parameters
        # Kalman gain, could be considered as a dynamic 'learning rate'
        self.kg = np.ones((kwargs['ncontext'], self.noption)) * kg0
        # mean of reward pdf
        self.mu = np.ones((kwargs['ncontext'], self.noption)) * mu0
        self.value = self.mu
        # variance
        self.v = np.ones((kwargs['ncontext'], self.noption)) * v0
        # innovation variance
        self.sig_xi = sig_xi
        # noise variance
        self.sig_eps = sig_eps
        # stochasticity
        self.beta = kwargs['beta']

    def update_pdf(self, s, a, r):
        """
        computes posterior mean and variance
        :param s:
        :param a:
        :param r:
        :return:
        """
        self.mu[s, a] = self.mu[s, a] + self.kg[s, a] * (r - self.mu[s, a])
        self.v[s, a] = (1 - self.kg[s, a]) * (self.v[s, a] + self.sig_xi)

    def update_kg(self, s, a):
        """
        updates the Kalman gain
        :param s: state
        :param a: action
        :return:
        """
        self.kg[s, a] = (
            self.v[s, a] + self.sig_xi + 1e-20) / (
                (self.v[s, a] + self.sig_xi + self.sig_eps) + 1e-20)

    def learn(self, s, a, r):
        self.update_kg(s, a)
        self.update_pdf(s, a, r)


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
        lb=.01, ub=0.95, noption=noption, tmax=tmax
    )

    regret = np.zeros((2, nagent, tmax))

    # Run
    for i in range(2):
        for n in range(nagent):
            if not i:
                agent = KalmanAgent(
                    ncontext=env.ncontext,
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
                    ncontext=env.ncontext,
                    noption=noption,
                    tmax=tmax,
                    q0=0,
                    alpha=alpha,
                    beta=beta2
                )

            for t in range(tmax):
                s = env.con[t]
                a = agent.make_choice(s)
                r = env.play(s, a)

                # only update if there's a new turn
                if t < tmax - 1:

                    agent.learn(
                        s=s,
                        a=a,
                        r=r
                    )

                    agent.remember(
                        r=r,
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
