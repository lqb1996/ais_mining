from matplotlib import pyplot as plt
import numpy as np
import os
import math


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
    # plt.xlim(106, 122)
    # plt.ylim(0, 22)
    plt.grid(alpha=0.2)
    # plt.legend(frameon=True, fontsize=5, loc='upper left')
    # plt.xlabel(x.name,fontsize=5)
    # plt.ylabel(y.name,fontsize=5)
    # plt.xlim(x.quantile(x_l),x.quantile(x_u))
    # plt.ylim(y.quantile(y_l),y.quantile(y_u))
    plt.savefig(file)


def cluster_plot(cluster_data, x_l=0, x_u=1, y_l=0, y_u=1, dir='png/cluster_res'):
    plt.figure(dpi=700, figsize=(4, 4))
    colors = ['pink', 'c', 'black']
    # categories=[2,0,1]
    labels = ['ground_truth', 'predict']
    alphas = [1, 0.35, 0.35]
    cls = cluster_data[:, 2]
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        x = cluster_data[idx, 0]
        y = cluster_data[idx, 1]
        plt.figure(dpi=700, figsize=(4, 4))
        plt.scatter(x,
                    y,
                    s=6,
                    c='pink',
                    # label=labels[0],
                    # edgecolors=colors[0],
                    alpha=0.3,
                    linewidths=0.01)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5, rotation=90)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(os.path.join(dir, 'cls_'+str(c)+'.png'))


def loss_plot(epoch, train_loss, val_loss, lr, lossfile='loss.png', lrfile='lr.png'):
    plt.figure(dpi=700, figsize=(12, 4))
    colors = ['pink', 'c', 'black']
    labels = ['train_loss', 'val_loss', 'lr']
    plt.subplot(2, 2, 1)
    plt.plot(epoch,
             train_loss,
             c=colors[0],
             label=labels[0]
             )
    plt.subplot(2, 2, 2)
    plt.plot(epoch,
             val_loss,
             c=colors[1],
             label=labels[1],
             )
    plt.subplot(2, 2, 3)
    plt.plot(epoch,
             lr,
             c=colors[2],
             label=labels[2],
             )
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5, rotation=90)
    # plt.xlim(0, 1000)
    # plt.ylim(0, 20)
    plt.grid(alpha=0.2)
    plt.legend(frameon=True, fontsize=5, loc='upper right')
    plt.xlabel('epoch', fontsize=5)
    plt.ylabel('loss', fontsize=5)
    plt.savefig(lossfile)


def muti_loss_plot(epoch, train_losses, loss_file='muti_loss.png'):
    plt.figure(dpi=700, figsize=(12, 4))
    colors = ['pink', 'c', 'black', 'blue', 'orange', 'red']
    for i, loss in enumerate(train_losses):
        e = epoch[:len(loss)]
        plt.plot(e,
                 loss,
                 c=colors[i % 6],
                 label=i
                 )
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5, rotation=90)
    plt.grid(alpha=0.2)
    plt.legend(frameon=True, fontsize=5, loc='upper right')
    plt.xlabel('epoch', fontsize=5)
    plt.ylabel('loss', fontsize=5)
    plt.savefig(loss_file)


def process_muti_file(file_list, type=0, use_log=True):
    epochs = []
    losses = []
    seg_str = ['train_loss : ', 'val_loss : ', 'lr : ', 'train_pre_loss : ', 'val_pre_loss : ']
    for fl in file_list:
        loss = []
        epoch = []
        with open(fl, 'r') as f:
            for l in f.readlines():
                epoch.append(int(l.split('epoch : ')[1].split('  ')[0]))
                loss.append(math.log(float(l.split(seg_str[type])[1].split('  ')[0]) if float(l.split(seg_str[type])[1].split('  ')[0]) > 0 else 0.00001) if use_log else float(l.split(seg_str[type])[1].split('  ')[0]))
            losses.append(loss)
            epochs.append(epoch)
    max_e = max([len(e) for i, e in enumerate(epochs)])
    epochs = np.arange(max_e)
    muti_loss_plot(epochs, losses)


if __name__ == '__main__':
    file_list = [
        # 'result/07-06_09:28/lstm4pre.log',
        'result/07-08_10:12/lstm4pre.log',
        # 'result/07-10_16:10/lstm4pre.log',
        # 'result/07-19_01:24/lstm4pre.log',
    ]
    process_muti_file(file_list, type=3, use_log=True)
