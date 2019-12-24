import numpy as np

from goat.utils.pareto_front import identify_pareto
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize


def main():
    problem = get_problem("zdt1")
    algorithm = NSGA2(pop_size=300, eliminate_duplicates=True)

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 200),
                   seed=1,
                   verbose=True)

    x = res.X
    y = res.F

    np.save('result_x.npy', x)
    np.save('result_y.npy', y)

    pareto_points = identify_pareto(-1 * y)
    x_pareto_points = x[pareto_points, ...]
    y_pareto_points = y[pareto_points, ...]

    np.save('result_x_pareto.npy', x_pareto_points)
    np.save('result_y_pareto.npy', y_pareto_points)


if __name__ == '__main__':
    main()
