import matplotlib.pyplot as plt
import numpy as np

from goat.test_functions.multi_objective import ZDT1


def main():
    plt.figure()

    problem = ZDT1()
    pareto_front = problem.pareto_front(n_pareto_points=100)
    f1_pf = pareto_front[:, 0]
    f2_pf = pareto_front[:, 1]
    plt.plot(f1_pf, f2_pf, 'k-', label="True Pareto Front")

    hyperopt_pf = np.load('./_hyperopt/result_y_pareto.npy')
    f1_hyperopt_pf = hyperopt_pf[:, 0]
    f2_hyperopt_pf = hyperopt_pf[:, 1]
    plt.scatter(f1_hyperopt_pf, f2_hyperopt_pf, label='TPE (Hyperopt)')

    mobo_nsga2_ei_pf = np.load('./_mobo_sode-nsga2-ei/result_y_pareto.npy')
    f1_mobo_nsga2_ei_pf = mobo_nsga2_ei_pf[:, 0]
    f2_mobo_nsga2_ei_pf = mobo_nsga2_ei_pf[:, 1]
    plt.scatter(f1_mobo_nsga2_ei_pf, f2_mobo_nsga2_ei_pf, label='NSGA2:EI (mobo)')

    mobo_nsga2_ucb_pf = np.load('./_mobo_sode-nsga2-ucb/result_y_pareto.npy')
    f1_mobo_nsga2_ucb_pf = mobo_nsga2_ucb_pf[:, 0]
    f2_mobo_nsga2_ucb_pf = mobo_nsga2_ucb_pf[:, 1]
    plt.scatter(f1_mobo_nsga2_ucb_pf, f2_mobo_nsga2_ucb_pf, label='NSGA2:UCB (mobo)')

    pymoo_nsga2_pf = np.load('_pymoo-nsga2/result_y_pareto.npy')
    f1_pymoo_nsga2_pf = pymoo_nsga2_pf[:, 0]
    f2_pymoo_nsga2_pf = pymoo_nsga2_pf[:, 1]
    plt.scatter(f1_pymoo_nsga2_pf, f2_pymoo_nsga2_pf, label='NSGA2 (pymoo)')

    bayes_opt_pf = np.load('_bayes_opt-ucb/result_y_pareto.npy')
    f1_bayes_opt_pf = bayes_opt_pf[:, 0]
    f2_bayes_opt_pf = bayes_opt_pf[:, 1]
    plt.scatter(f1_bayes_opt_pf, f2_bayes_opt_pf, label='L-BFGS-B:UCB (bayes-opt)')

    platypus_pf = np.load('_Platypus/result_y_pareto.npy')
    f1_platypus_pf = platypus_pf[:, 0]
    f2_platypus_pf = platypus_pf[:, 1]
    plt.scatter(f1_platypus_pf, f2_platypus_pf, label='NSGA2 (Platypus)')

    jmetalpy_pf = np.load('_jmetalpy/result_y_pareto.npy')
    f1_jmetalpy_pf = jmetalpy_pf[:, 0]
    f2_jmetalpy_pf = jmetalpy_pf[:, 1]
    plt.scatter(f1_jmetalpy_pf, f2_jmetalpy_pf, label='NSGA2 (jmetalpy)')

    plt.xlabel('f1(x)')
    plt.ylabel('f2(x)')
    plt.legend()

    plt.savefig('result.png')

    plt.show()


if __name__ == '__main__':
    main()
