import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import tensorflow as tf  # noqa

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
    color_list = ['tab:blue', 'tab:orange']
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

        plt.pause(0.0001)

        if iteration != range(iterations - 1)[-1]:
            for scatter in scatter_list:
                scatter.remove()
            for text in text_list:
                text.remove()

    plt.pause(10)


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


def main():
    x_0 = 0.75
    y_0 = 1.0
    func_name = 'kaist'
    target = -1

    ax = draw_cost_function(func_name=func_name)
    sgd_history = optimize_tf(x_0, y_0, tf.keras.optimizers.SGD(lr=0.1), func_name=func_name, target=target)
    adam_history = optimize_tf(x_0, y_0, tf.keras.optimizers.Adam(lr=0.1), func_name=func_name, target=target)
    draw_optimize_history(ax, [sgd_history, adam_history], ['SGD (lr=0.1)', 'Adam (lr=0.1)'])


if __name__ == '__main__':
    main()
