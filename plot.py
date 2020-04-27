from matplotlib import pyplot as plt


def plot(x,y,x_l=0,x_u=1,y_l=0,y_u=1):
    plt.figure(dpi=700,figsize=(4,4))
    colors = ['pink', 'c', 'black']
    # categories=[2,0,1]
    # labels=['拖网','围网','刺网']
    alphas = [1, 0.65, 0.35]
    for i in range(3):
        plt.scatter(x,
                    y,
                    s=6,
                    c=colors[i],
                    # label=labels[i],
                    edgecolors=colors[i],
                    alpha=alphas[i],
                    linewidths=0.01)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5,rotation=90)
    plt.grid(alpha=0.2)
    # plt.legend(frameon=True,fontsize=5,loc='upper left')
    plt.xlabel(x.name,fontsize=5)
    plt.ylabel(y.name,fontsize=5)
    plt.xlim(x.quantile(x_l),x.quantile(x_u))
    plt.ylim(y.quantile(y_l),y.quantile(y_u))
    plt.show()
