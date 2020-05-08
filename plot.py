from matplotlib import pyplot as plt


def plot(x, y, x_pre, y_pre, x_l=0, x_u=1, y_l=0, y_u=1, file='test.png'):
    plt.figure(dpi=700, figsize=(4, 4))
    colors = ['pink', 'c', 'black']
    # categories=[2,0,1]
    labels = ['ground_truth', 'predict']
    alphas = [1, 0.35, 0.35]
    plt.scatter(x,
                y,
                s=6,
                c=colors[0],
                label=labels[0],
                edgecolors=colors[0],
                alpha=alphas[0],
                linewidths=0.01)
    plt.scatter(x_pre,
                y_pre,
                s=6,
                c=colors[1],
                label=labels[1],
                edgecolors=colors[1],
                alpha=alphas[1],
                linewidths=0.01)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5, rotation=90)
    plt.grid(alpha=0.2)
    # plt.legend(frameon=True, fontsize=5, loc='upper left')
    # plt.xlabel(x.name,fontsize=5)
    # plt.ylabel(y.name,fontsize=5)
    # plt.xlim(x.quantile(x_l),x.quantile(x_u))
    # plt.ylim(y.quantile(y_l),y.quantile(y_u))
    plt.savefig(file)
