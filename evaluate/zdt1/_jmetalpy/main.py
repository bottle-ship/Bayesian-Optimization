import numpy as np

from goat.utils.pareto_front import identify_pareto
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations


def main():
    problem = ZDT1()

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()

    x = list()
    y = list()
    for front in algorithm.get_result():
        x.append(front.variables)
        y.append(front.objectives)

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
