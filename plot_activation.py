import matplotlib.pyplot as plt
import numpy as np
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relun(x, n):
    # n = args.n
    # return np.min(np.max(0, x), n)
    return [0 if x_ < 0 else n if x_ > n else x_ for x_ in x]

def prelu(x, args):
    n = args.alpha
    return [n*x_ if x_ < 0 else x_ for x_ in x]

def activation(x):
    return relun(x)

def plot_sigmoid(args):
    x = np.linspace(-10, 10, 1000)
    y = relun(x, 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, linewidth=2, color='red', label='n=1')
    
    y = relun(x, 5)
    ax.plot(x, y, linewidth=2, color='blue', label='n=5')
    
    y = relun(x, 7)
    ax.plot(x, y, linewidth=2, color='green', label='n=7')
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)

    ax.tick_params(axis='both', which='both', direction='in', length=6, width=0.5)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_title('PReLU Activation Function', fontsize=16)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    # plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('relu_2.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n', default=3)
    parser.add_argument('--alpha', default=0.2)
    
    args = parser.parse_args()
    plot_sigmoid(args)