from matplotlib import pyplot as plt


def line_plot(line1_X, line1_Y, line2_X, line2_Y, path, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1_X, line1_Y, label=label1, linewidth=lw)
    ax.plot(line2_X, line2_Y, label=label2, linewidth=lw)
    ax.set_ylabel('Price', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    fig.savefig(path)
