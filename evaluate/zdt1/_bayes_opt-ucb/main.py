import numpy as np

from bayes_opt import BayesianOptimization
from goat.test_functions.multi_objective import ZDT1
from goat.utils.pareto_front import identify_pareto


def objective(*args, **kwargs):
    x = list()
    for i in range(1, 31):
        x.append(kwargs['x%02d' % i])
    x = np.array([x])

    problem = ZDT1()
    y = problem.evaluate(x)

    loss = np.sum(y)

    return -1 * loss


def main():
    max_evals = 200

    pbounds = dict()
    for i in range(1, 31):
        pbounds['x%02d' % i] = (0.0, 1.0)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        acq='ucb'
    )
    optimizer.maximize(init_points=64, n_iter=max_evals)

    x = list()
    for res in optimizer.res:
        params = res['params']
        data = list()
        for i in range(1, 31):
            data.append(params['x%02d' % i])
        x.append(np.array([data.copy()]))
    x = np.vstack(x)
    problem = ZDT1()
    y = problem.evaluate(x)

    np.save('result_x.npy', x)
    np.save('result_y.npy', y)

    pareto_points = identify_pareto(-1 * y)
    x_pareto_points = x[pareto_points, ...]
    y_pareto_points = y[pareto_points, ...]

    np.save('result_x_pareto.npy', x_pareto_points)
    np.save('result_y_pareto.npy', y_pareto_points)


if __name__ == "__main__":
    main()
