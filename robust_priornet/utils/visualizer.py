import os
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_curve(x, y, axis, x_label='', y_label='',
               title='', curve_title='',
               x_lim=None, y_lim=None,
               show_legend=False,
               axis_spine_visibility_config: list = None):
    """
        Plots a line graph with x values on x-axis and y values on y-axis. 

        Args:
            axis: axis of the subplot (already indexed to right subplot)
            x_label: str value
            y_label: str value
            curve_title: str value to be used for the line plot specific to given x,y values.
            title: str value representing subplot title
            x_lim: tuple representing low and high range for x-axis
            y_lim: tuple representing low and high range for y-axis
            axis_spine_visibility_config: list of axis directions which should be made invisible.
    """
    if axis_spine_visibility_config is not None:
        for spine in axis_spine_visibility_config:
            fetched_spine = axis.spines[spine]
            fetched_spine.set_visible(False)

    axis.set_title("\n".join(wrap(title, 20)))
    axis.plot(x, y, label=curve_title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    if x_lim is not None:
        axis.set_xlim(*x_lim)
    if y_lim is not None:
        axis.set_ylim(*y_lim)
    if show_legend is True:
        axis.legend()

def get_epoch_summaries(model_dir, num_epochs, verbose=False):
    """
    Accumulates the loss, accuracy values for in-domain and out-of-domain samples
    across training, validation phases. Uses the summary files written by
    robust_priornet.training.trainer.
    """
    train_id_loss = []
    train_ood_loss = []
    val_id_loss = []
    val_ood_loss = []
    train_id_accuracy = []
    val_id_accuracy = []
    for epoch in range(num_epochs):
        summary = torch.load(os.path.join(model_dir, f'epoch_summary_{epoch+1}.pt'))
        if verbose:
            print(summary)
        train_id_loss.append(summary['train_results']['id_loss'])
        train_ood_loss.append(summary['train_results']['ood_loss'])
        train_id_accuracy.append(summary['train_results']['id_accuracy'])
        val_id_loss.append(summary['val_results']['id_loss'])
        val_ood_loss.append(summary['val_results']['ood_loss'])
        val_id_accuracy.append(summary['val_results']['id_accuracy'])

    return train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy

def plot_loss_accuracy_curve(model_dir, num_epochs):
    train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy = get_epoch_summaries(model_dir, num_epochs)
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # ID axis for loss curve
    x_values = np.arange(num_epochs)
    plot_curve(x_values, train_id_loss, axes[0,0], x_label='Epochs',
               y_label='Loss', title='Loss curve',
               curve_title='ID - Train loss', show_legend=True)
    plot_curve(x_values, val_id_loss, axes[0,0], x_label='Epochs',
               y_label='Loss', title='Loss curve',
               curve_title='ID - Validation loss', show_legend=True)

    # OOD axis for loss curve
    plot_curve(x_values, train_ood_loss, axes[0,1], x_label='Epochs',
               y_label='Loss', title='Loss curve',
               curve_title='OOD - Train loss', show_legend=True)
    plot_curve(x_values, val_ood_loss, axes[0,1], x_label='Epochs',
               y_label='Loss', title='Loss curve',
               curve_title='OOD - Validation loss', show_legend=True)

    # ID axis for accuracy curve
    plot_curve(x_values, train_id_accuracy, axes[1,0], x_label='Epochs',
               y_label='Accuracy', title='Accuracy curve',
               curve_title='ID - Train accuracy', show_legend=True)
    plot_curve(x_values, val_id_accuracy, axes[1,0], x_label='Epochs',
               y_label='Accuracy', title='Accuracy curve',
               curve_title='ID - Validation accuracy', show_legend=True)

    figure.tight_layout()
    plt.show()
    
def plot_aupr_auroc(aupr_list, auroc_list):
    print("AU_PR: ", aupr_list)
    print("AU_ROC: ", auroc_list)
    x = np.arange(len(aupr_list)) 
    width = 0.35
    auprs = [np.round(x * 100.0, 1) for x in aupr_list]
    aurocs = [np.round(x * 100.0, 1) for x in auroc_list]

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, auprs, width, label='AU_PR')
    rects2 = ax.bar(x + width/2, aurocs, width, label='AU_ROC')

    ax.set_ylabel('AC')
    ax.set_title('AC by different models')
    ax.set_xticks(x)
    ax.set_xticklabels([f'model{i}' for i in range(len(aupr_list))])
    ax.legend()

    fig.tight_layout()
    plt.show()
