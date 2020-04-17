import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'svg.fonttype': 'none', 'font.size': 14})
from matplotlib import pyplot as plt

def main():
    data = np.loadtxt("velocity.csv", delimiter=",")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(data.shape[0]):
        ax.plot(data[i, :], color="grey", alpha=0.7, lw=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Spatial position")
    ax.set_ylabel("$u(x, t)$")
    plt.show()

if __name__ == '__main__':
    main()