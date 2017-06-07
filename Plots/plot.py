import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def read_data(fname):
    # assumes that file fname has three columns: Wall Time, Step, Value
    d = [[], [], []]
    with open(fname, 'r') as f:
        f.readline() # to read headers
        for line in f:
            line = [float(x) for x in line.strip().split(',')]
            d[0].append(line[0])
            d[1].append(line[1])
            d[2].append(line[2])
    return d

def plot(data_dir, x_axis, title, names):
    ## choose files to plot and read their data
    fnames = glob.glob(os.path.join(data_dir, '*csv'))
    data = [read_data(fname) for fname in fnames]

    ## colors
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

    ## sanity checks
    if x_axis not in ['Wall Time', 'Step']:
        print 'Invalid x_axis type. Exiting.'
        exit()
    
    ## plot
    plot_fill(data, colors, x_axis, title, names)


def plot_fill(data, colors, x_axis, title, names):
    ax = plt.figure()

    # plot data
    for i in range(len(data)):
        if x_axis=='Wall Time':
            plt.plot([data[i][0][j]-data[i][0][0] for j in range(len(data[i][0]))], data[i][2], linewidth=5, color=colors[i])
        elif x_axis=='Step':
            plt.plot([data[i][1][j]-data[i][0][0] for j in range(len(data[i][1]))], data[i][2], linewidth=5, color=colors[i])

    # details
    plt.tick_params(labelsize=20)
    plt.grid()
    legend = plt.legend(names, fontsize=24)
    for label in legend.get_lines():
        label.set_linewidth(20)
    plt.xlabel(x_axis, fontsize=24)
    plt.ylabel(y_axis, fontsize=24)
    plt.title(title, fontsize=30)
    plt.show()


if __name__=='__main__':
    data_dir = 'data_dir1'
    x_axis = 'Wall Time' # one of Wall-Time or Step
    y_axis = 'Mean Score'
    title = 'Testing'
    names = ['1', '2'] # legend names. Name of each line that we are plotting
    plot(data_dir, x_axis, title, names=names)
