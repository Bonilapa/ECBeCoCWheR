import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, y, figure_file, title, avg = True):
    plt.clf()
    if avg:
        running_avg = np.zeros(len(y))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(y[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
    else:
        plt.plot(x,y)
    plt.title(title)
    plt.savefig(figure_file)


def plot_learning_curve_2(x, y, label1, y1, label2, figure_file, title):

    # for i in range(len(y1)):
    #     print(y1[i])
    plt.clf()
    running_avg_y = np.zeros(len(x))
    for i in range(len(running_avg_y)):
        running_avg_y[i] = y[i]#np.mean(y[max(0, i-100):(i+1)])
    running_avg_y1 = np.zeros(len(x))
    for i in range(len(running_avg_y1)):
        running_avg_y1[i] = y1[i]#np.mean(y1[max(0, i-100):(i+1)])
    plt.plot(x, running_avg_y,"-", running_avg_y1, "--")
    # plt.plot(y, running_avg)
    plt.title(title)
    plt.legend([label1, label2])
    plt.savefig(figure_file)