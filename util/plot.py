import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, path, info=None):
    fig = plt.figure(figsize=(14, 7))
    heatmap = plt.pcolor(alignment, cmap=matplotlib.cm.Blues)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close(fig)
