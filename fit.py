from main import Environment, Agent, KalmanAgent, QLearningAgent


def get_parameters(force, parameters_file='data/params.p'):
    import pickle

    if force:
        sig_eps, sig_xi, beta1, alpha, beta2 = optimize()
        pickle.dump(
            file=open(parameters_file, 'wb'),
            obj=[sig_eps, sig_xi, beta1, alpha, beta2]
        )

    else:
        try:
            sig_eps, sig_xi, beta1, alpha, beta2 = \
                pickle.load(open(parameters_file, 'rb'))

        except FileNotFoundError:
            sig_eps, sig_xi, beta1, alpha, beta2 = optimize()
            pickle.dump(
                file=open(parameters_file, 'wb'),
                obj=[sig_eps, sig_xi, beta1, alpha, beta2]
            )

    return sig_eps, sig_xi, beta1, alpha, beta2


def optimize():
    import pyfmincon.opt

    # run matlab engine
    pyfmincon.opt.start()

    options = {
        #'Algorithm': 'interior-point',
        #'Display': 'off',
        'MaxIter': 20000,
        'MaxFunEval': 20000
    }

    [sig_eps, sig_xi, beta1], ll1, exitflag1 = pyfmincon.opt.fmincon(
        'fit.run_optimization',
        x0=[0, 0, 1],
        lb=[0, 0, 0],
        ub=[1, 1, 300],
        options=options)

    [alpha, beta2], ll2, exitflag2 = pyfmincon.opt.fmincon(
        'fit.run_optimization',
        x0=[0.5, 0],
        lb=[0, 0],
        ub=[1, 300],
        options=options)

    pyfmincon.opt.stop()

    return sig_eps, sig_xi, beta1, alpha, beta2


def run_optimization(args, noption=5, tmax=1000):

    env = Environment(lb=.01, ub=0.95, noption=noption)

    if len(args) == 3:
        agent = KalmanAgent(
            noption=noption,
            tmax=tmax,
            kg0=0,
            mu0=0,
            v0=0,
            sig_eps=args[0],
            sig_xi=args[1],
            beta=args[2]
        )
    else:
        agent = QLearningAgent(
            noption=noption,
            tmax=tmax,
            q0=0,
            alpha=args[0],
            beta=args[1]
        )

    for t in range(tmax):
        choice = agent.make_choice()
        reward = env.play(choice)

        # only update if there's a new turn
        if t < tmax - 1:
            agent.learn(choice, reward)
            agent.remember(reward=reward, choice=choice, t=t)

    return agent.regret[-1]


if __name__ == '__main__':
    exit('Please run the main.py script.')
