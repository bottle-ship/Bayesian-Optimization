import numpy as np

from goat.utils.pareto_front import identify_pareto
from mobo_sode.optimizer import NSGA2
from mobo_sode.model import ExactGPModel
from mobo_sode.acquisition import ei
from mobo_sode.bayesopt import MultiObjectiveBayesianOpt
from mobo_sode.test_functions import zdt1


if __name__ == "__main__":
    # multi objective genetic algorithm (NSGA2) is implemented with 'DEAP'
    # Gaussian Process model is implemented with 'gpytorch'
    opt = MultiObjectiveBayesianOpt(
        evaluation_function=zdt1,
        surrogate_model=ExactGPModel,
        optimizer=NSGA2,
        acquisition=ei,
        n_objective_dimension=2,
        n_design_variables_dimension=30,
        n_initial_sample=64,
        n_new_samples=8,
        bayesian_optimization_iter_max=20,
        likelihood_optimization_iter_max=5000,
        likelihood_optimization_criteria=1e-6,
        n_ga_population=16,
        n_ga_generation=100,
    )
    result = opt.optimize()

    x = result[0].numpy()
    y = result[1].numpy()

    np.save('result_x.npy', x)
    np.save('result_y.npy', y)

    pareto_points = identify_pareto(-1 * y)
    x_pareto_points = x[pareto_points, ...]
    y_pareto_points = y[pareto_points, ...]

    np.save('result_x_pareto.npy', x_pareto_points)
    np.save('result_y_pareto.npy', y_pareto_points)
