import numpy as np

from goat.utils.pareto_front import identify_pareto
from platypus import ZDT1, NSGAII


def main():
    problem = ZDT1()
    algorithm = NSGAII(problem)
    algorithm.run(10000)

    x = list()
    y = list()
    for solution in algorithm.result:
        x.append(solution.variables)
        y.append(solution.objectives)

    x = np.array(x)
    y = np.array(y)

    np.save('result_x.npy', x)
    np.save('result_y.npy', y)

    pareto_points = identify_pareto(-1 * y)
    x_pareto_points = x[pareto_points, ...]
    y_pareto_points = y[pareto_points, ...]

    np.save('result_x_pareto.npy', x_pareto_points)
    np.save('result_y_pareto.npy', y_pareto_points)


if __name__ == '__main__':
    main()
