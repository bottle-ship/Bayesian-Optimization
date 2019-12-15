import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import tensorflow as tf  # noqa

from bayes_opt import BayesianOptimization  # noqa
from functools import partial  # noqa
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa


def cost_function(x, y, func_name='kaist', return_np=False):
    if func_name == 'kaist':
        z = -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))
    elif func_name == 'sphere':
        z = tf.square(x) + tf.square(y)
    elif func_name == 'rosenbrock':
        z = 100 * tf.square(y - tf.square(x)) + tf.square(1 - x)
    else:
        raise ValueError()

    if return_np:
        return z.numpy()
    else:
        return z


def bayes_opt_objective(x, y, func_name='kaist', target=None):
    z = cost_function(x, y, func_name=func_name, return_np=True)
    if target is None:
        loss = z
    else:
        loss = np.abs(z - target)

    return -1 * loss


def hyperopt_objective(space):
    x = space['x']
    y = space['y']
    func_name = space['func_name']
    target = space['target']

    z = cost_function(x, y, func_name=func_name, return_np=True)
    if target is None:
        loss = z
    else:
        loss = np.abs(z - target)

    return {'loss': loss, 'status': STATUS_OK, 'history': [x, y, z, loss]}


def draw_cost_function(func_name='sphere'):
    plt.ion()
    fig = plt.figure(figsize=(3, 2), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    params = {'legend.fontsize': 3,
              'legend.handlelength': 3}
    plt.rcParams.update(params)
    plt.axis('off')

    x_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
    y_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
    x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
    x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
    y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
    z_val_mesh_flat = cost_function(x_val_mesh_flat, y_val_mesh_flat, func_name=func_name, return_np=True)
    z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)

    ax.plot_surface(x_val_mesh, y_val_mesh, z_val_mesh, alpha=.4, cmap='coolwarm')
    plt.draw()

    return ax


def draw_optimize_history(ax, history, label):
    color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    iterations = len(history[0])

    for iteration in range(iterations - 1):
        scatter_list = list()
        text_list = list()
        for i in range(len(history)):
            x_pre = history[i][iteration][0]
            y_pre = history[i][iteration][1]
            z_pre = history[i][iteration][2]

            x_val = history[i][iteration + 1][0]
            y_val = history[i][iteration + 1][1]
            z_val = history[i][iteration + 1][2]

            loss = history[i][iteration + 1][3]

            ax.plot([x_pre, x_val], [y_pre, y_val], [z_pre, z_val],
                    linewidth=0.5, color=color_list[i])
            scatter_list.append(ax.scatter(x_val, y_val, z_val,
                                           s=1, depthshade=True, color=color_list[i]))
            text_list.append(ax.text(x_val, y_val, z_val,
                                     '  x: {:.2f}, y: {:.2f}, z: {:.2f}, cost: {:.2f}'.format(
                                         x_val, y_val, z_val, loss
                                     ), fontsize=3))

        if iteration == 0:
            plt.legend(scatter_list, label)

        plt.pause(0.001)

        if iteration != range(iterations - 1)[-1]:
            for scatter in scatter_list:
                scatter.remove()
            for text in text_list:
                text.remove()

    plt.pause(100)


def optimize_tf(x_0, y_0, optimizer, func_name='kaist', target=None, iterations=100):
    z_0 = cost_function(x_0, y_0, func_name=func_name, return_np=True)

    history = [[x_0, y_0, z_0, np.nan]]

    x = tf.Variable(x_0, dtype=tf.float32)
    y = tf.Variable(y_0, dtype=tf.float32)

    for iteration in range(iterations):
        with tf.GradientTape() as tape:
            z = cost_function(x, y, func_name=func_name, return_np=False)
            if target is None:
                loss = z
            else:
                loss = tf.abs(z - target)
        grad = tape.gradient(loss, [x, y])
        optimizer.apply_gradients(zip(grad, [x, y]))

        history.append([x.numpy(), y.numpy(), z.numpy(), loss.numpy()])

    return history


def opimize_hyperopt(x_0, y_0, func_name='kaist', target=None, iterations=100):
    space = dict()
    space['x'] = hp.uniform('x', -1.5, 1.5)
    space['y'] = hp.uniform('y', -1.5, 1.5)
    space['func_name'] = func_name
    space['target'] = target

    trials = Trials()
    fmin(hyperopt_objective, space=space, algo=tpe.suggest, max_evals=iterations, trials=trials)

    history = [[x_0, y_0, cost_function(x_0, y_0, func_name=func_name, return_np=True), np.nan]]
    for i in range(iterations):
        history.append(trials.results[i]['history'])

    return history


def optimize_bayes_opt(x_0, y_0, func_name='kaist', target=None, iterations=100):
    f = partial(bayes_opt_objective, func_name=func_name, target=target)
    pbounds = {'x': (-1.5, 1.5), 'y': (-1.5, 1.5)}

    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds
    )
    optimizer.maximize(init_points=0, n_iter=iterations)

    history = [[x_0, y_0, cost_function(x_0, y_0, func_name=func_name, return_np=True), np.nan]]
    for i, res in enumerate(optimizer.res):
        params = res['params']
        cost = res['target']
        x = params['x']
        y = params['y']
        z = cost_function(x, y, func_name=func_name, return_np=True)

        history.append([x, y, z, cost])

    return history


def main():
    x_0 = 0.75
    y_0 = 1.0
    func_name = 'kaist'
    target = -1
    iterations = 100

    ax = draw_cost_function(func_name=func_name)
    sgd_history = optimize_tf(
        x_0, y_0, tf.keras.optimizers.SGD(lr=0.1), func_name=func_name, target=target, iterations=iterations
    )
    adam_history = optimize_tf(
        x_0, y_0, tf.keras.optimizers.Adam(lr=0.1), func_name=func_name, target=target, iterations=iterations
    )
    bayes_opt_history = optimize_bayes_opt(x_0, y_0, func_name=func_name, target=target, iterations=iterations)
    hyperopt_history = opimize_hyperopt(x_0, y_0, func_name=func_name, target=target, iterations=iterations)
    draw_optimize_history(ax,
                          [sgd_history, adam_history, bayes_opt_history, hyperopt_history],
                          ['SGD (lr=0.1)', 'Adam (lr=0.1)', 'Bayes-GP', 'Bayes-TPE'])


if __name__ == '__main__':
    main()
