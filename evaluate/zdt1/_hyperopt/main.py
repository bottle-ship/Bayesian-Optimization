import numpy as np

from goat.test_functions.multi_objective import ZDT1
from goat.utils.pareto_front import identify_pareto
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


def objective(space):
    x = list()
    for i in range(1, 31):
        x.append(space['x%02d' % i])
    x = np.array([x])

    problem = ZDT1()
    y = problem.evaluate(x)

    loss = np.sum(y)

    return {'loss': loss, 'status': STATUS_OK, 'x': x, 'y': y}


def main():
    max_evals = 5000

    space = dict()
    for i in range(1, 31):
        space['x%02d' % i] = hp.uniform('x%02d' % i, 0, 1)

    trials = Trials()
    fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    x = list()
    y = list()
    for i in range(max_evals):
        x.append(trials.results[i]['x'])
        y.append(trials.results[i]['y'])

    x = np.vstack(x)
    y = np.vstack(y)

    np.save('result_x.npy', x)
    np.save('result_y.npy', y)

    pareto_points = identify_pareto(-1 * y)
    x_pareto_points = x[pareto_points, ...]
    y_pareto_points = y[pareto_points, ...]

    np.save('result_x_pareto.npy', x_pareto_points)
    np.save('result_y_pareto.npy', y_pareto_points)


if __name__ == "__main__":
    main()
