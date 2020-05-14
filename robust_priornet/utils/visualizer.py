import os
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from ..eval.uncertainty import UncertaintyMeasuresEnum

def plot_curve(x, y, axis, x_label='', y_label='',
               title='', curve_title='',
               x_lim=None, y_lim=None,
               show_legend=False,
               axis_spine_visibility_config: list = None,
               additional_plt_args=[]):
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
    axis.plot(x, y, *additional_plt_args, label=curve_title)
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
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        summary = torch.load(os.path.join(model_dir, f'epoch_summary_{epoch+1}.pt'))
        if verbose:
            print(summary)
        train_id_loss.append(summary['train_results']['id_loss'])
        train_ood_loss.append(summary['train_results']['ood_loss'])
        train_loss.append(summary['train_results']['loss'])
        train_id_accuracy.append(summary['train_results']['id_accuracy'])
        val_id_loss.append(summary['val_results']['id_loss'])
        val_ood_loss.append(summary['val_results']['ood_loss'])
        val_loss.append(summary['val_results']['loss'])
        val_id_accuracy.append(summary['val_results']['id_accuracy'])

    return train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy, train_loss, val_loss

def plot_loss_accuracy_curve(model_dir, num_epochs):
    train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy, train_loss, val_loss = get_epoch_summaries(model_dir, num_epochs)
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

    # Overall loss axis
    plot_curve(x_values, train_loss, axes[1,1], x_label='Epochs',
               y_label='Loss', title='Loss curve',
               curve_title='Train loss', show_legend=True)
    plot_curve(x_values, val_loss, axes[1,1], x_label='Epochs',
               y_label='Loss', title='Loss curve',
               curve_title='Validation loss', show_legend=True)
    
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

def plot_adv_samples(org_eval_dir, attack_dir, epsilon, plots_dir='vis',
                     known_misclassified_indices=[], limit=None):
    """
    Params
    ------
        org_eval_dir: directory where the model was evaluated for misclassification_detect task.
                        on the same dataset as the attack_dir
        attack_dir: directory where the various epsilon attack images and eval results are stored.
                        on the same dataset as org_eval_dir
        epsilon: value of the epsilon to locate the folder within the attack_dir.
        plots_dir: directory name to be created to store the results within the attack_dir.
        limit: number of adversarial samples to plot, otherwise all adversarial samples
                will be plotted.
    Returns
    -------
        adv_success: number of previously correctly classified samples that got misclassified under attack.
    """
    target_epsilon_dir = os.path.join(attack_dir, f"e{epsilon}-attack")
    probs = np.loadtxt(f"{target_epsilon_dir}/eval/probs.txt")
    labels = np.loadtxt(f"{target_epsilon_dir}/eval/labels.txt")
    # current confidence on attack images
    new_confidence = np.loadtxt(f"{target_epsilon_dir}/eval/confidence.txt")

    # misclassified samples under attack
    preds = np.argmax(probs, axis=1)
    misclassification = np.asarray(preds != labels, dtype=np.int32)
    misclassified = np.argwhere(misclassification == 1)
    print("# Misclassified samples under attack: ", misclassified.size)

    # real adversarial samples - original model correctly classified them, but now misclassified!
    # prob dist outputed by model on non-perturbed images.
    old_probs = np.loadtxt(f"{org_eval_dir}/id_probs.txt")
    # confidence of all attack_images from normal eval phase.
    old_confidence = np.loadtxt(f"{org_eval_dir}/id_confidence.txt")

    old_preds = np.argmax(old_probs, axis=1)
    correct_classifications = np.asarray(old_preds == labels, dtype=np.int32)
    correct_classified_indices = np.argwhere(correct_classifications == 1)
    print("# Correct classified samples prior attack: ", len(correct_classified_indices))

    misclassified = np.intersect1d(misclassified, correct_classified_indices)
    print("# Real adversarial samples under attack: ", misclassified.size)

    if len(known_misclassified_indices) > 0: # reduce to already known indices
        misclassified = np.intersect1d(known_misclassified_indices, misclassified)

    # create a separate dir to store all visualizations
    os.makedirs(os.path.join(target_epsilon_dir, plots_dir))

    # first plot
    figure, axes = plt.subplots(nrows = 10, ncols=3, figsize=(15, 15))
    figure.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, index in enumerate(misclassified):
        if index.ndim > 0:
            index = index[0]
        ri = i%10

        # org image
        axis = axes[ri][0]
        img = Image.open(f"{attack_dir}/org-images/{index}.png")
        axis.set_title(f"index: {index}, label: {int(labels[index])}, confidence: {np.round(old_confidence[index],3)}", pad=2)
        #hide all spines
        axis.axis("off")
        axis.imshow(img, cmap="gray")

        # adv image
        axis = axes[ri][1]
        img = Image.open(f"{target_epsilon_dir}/adv-images/{index}.png")
        axis.set_title(f"label: {preds[index]}, confidence: {np.round(new_confidence[index],3)}", pad=2)
        #hide all spines
        axis.axis("off")
        axis.imshow(img, cmap="gray")

        # change in prob dist (bar plot)
        axis = axes[ri][2]
        class_labels = np.arange(10)
        width = 0.2
        rects1 = axis.bar(class_labels - width/2, old_probs[index, :], width, label='Org')
        rects2 = axis.bar(class_labels + width/2, probs[index, :], width, label='Adv')
        axis.set_xticks(class_labels)
        axis.set_xticklabels(class_labels)
        # places the legend to the right of the current axis
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # batch every 10 sample into a single image
        if (i > 0 and (i+1) % 10 == 0) or i == (misclassified.size-1):
            plt.savefig(os.path.join(target_epsilon_dir, plots_dir,
                                     f"result_vis_{i+1}.png"), bbox_inches='tight')
            plt.close()

        if i > 0 and (i+1) % 10 == 0 and i != (misclassified.size-1):
            figure, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 15))
            figure.subplots_adjust(hspace=0.5, wspace=0.5)

        if limit is not None and i == limit-1:
            plt.close()
            break

    return misclassified.size

def plot_epsilon_curve(epsilon: list, adv_success_rates: list,
                       result_dir: str = '.',
                       file_name: str = 'epsilon-curve.png',
                       plt_label: str = '',
                       title: str='',
                       save_fig=True,
                       plt_axis=None):
    if plt_axis is None:
        _, plt_axis = plt.subplots(nrows=1, ncols=1)

    plt_axis.set_yticks(np.arange(0, 1.1, step=0.1))
    plt_axis.set_xticks(np.arange(np.min(epsilon), np.max(epsilon)+0.1, step=0.1))
    plot_curve(epsilon, adv_success_rates, plt_axis, x_label='Epsilon',
               y_label='Adversarial Success Rate',
               x_lim=(0.0, np.max(epsilon)+ 0.1), y_lim=(0.0, 1.1),
               curve_title=plt_label, show_legend=(True if plt_label != '' else False),
               title=f'Adversarial Success Rate vs Epsilon -{title}',
               additional_plt_args=['*-'])
    if save_fig:
        plt.savefig(os.path.join(result_dir, file_name))

def plot_many_epsilon_curves(epsilon: list, adv_success_rates: list,
                             curve_legend_labels: list,
                             plot_title: str,
                             result_dir: str,
                             file_name: str):
    # send adv_success_rates as list of lists for each curve to be plotted.
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    num_curves = len(adv_success_rates)
    for curve_index in range(num_curves):
        plot_epsilon_curve(epsilon, adv_success_rates[curve_index],
                           plt_label=curve_legend_labels[curve_index],
                           save_fig=False, plt_axis=axes, title=plot_title)
    plt.savefig(os.path.join(result_dir, file_name))

def plot_all_pr_curves(epsilons: list, src_attack_dir: str,
                       eval_dir_name: str, uncertainty_measure: UncertaintyMeasuresEnum,
                       result_dir: str):
    _, axes = plt.subplots(nrows=1, ncols=1)
    for epsilon in epsilons:
        target_epsilon_dir = os.path.join(src_attack_dir, f'e{epsilon}-attack')
        precision = np.loadtxt(os.path.join(target_epsilon_dir,
                                            eval_dir_name,
                                            f'{uncertainty_measure._value_}_precision.txt'))
        recall = np.loadtxt(os.path.join(target_epsilon_dir,
                                         eval_dir_name,
                                         f'{uncertainty_measure._value_}_recall.txt'))
        plot_curve(recall, precision, axes, x_label='Recall',
                   y_label='Precision',
                   x_lim=(0.0, 1.0), y_lim=(0.0, 1.0),
                   axis_spine_visibility_config=['right', 'top'],
                   curve_title=f'e:{epsilon}')
    
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(result_dir, f'{uncertainty_measure._value_}_PR_summary.png'),
                bbox_inches='tight')
    plt.close()

def plot_all_roc_curves(epsilons: list, src_attack_dir: str,
                        eval_dir_name: str, uncertainty_measure: UncertaintyMeasuresEnum,
                        result_dir: str):
    _, axes = plt.subplots(nrows=1, ncols=1)
    for epsilon in epsilons:
        target_epsilon_dir = os.path.join(src_attack_dir, f'e{epsilon}-attack')
        tpr = np.loadtxt(os.path.join(target_epsilon_dir,
                                      eval_dir_name,
                                      f'{uncertainty_measure._value_}_tpr.txt'))
        fpr = np.loadtxt(os.path.join(target_epsilon_dir,
                                      eval_dir_name,
                                      f'{uncertainty_measure._value_}_fpr.txt'))
        plot_curve(fpr, tpr, axes, x_label='False Postive Rate (FPR)',
                   y_label='True Positive Rate (TPR)',
                   x_lim=(0.0, 1.0), y_lim=(0.0, 1.0),
                   axis_spine_visibility_config=['right', 'top'],
                   curve_title=f'e:{epsilon}')
    
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(result_dir, f'{uncertainty_measure._value_}_ROC_summary.png'),
                bbox_inches='tight')
    plt.close()

def plot_adv_samples_ood(org_eval_dir, attack_dir, epsilon, threshold,
                         decision_measure: UncertaintyMeasuresEnum,
                         plots_dir='vis', limit=None):
    # old (no attack)
    old_id_uncertainty = np.loadtxt(f"{org_eval_dir}/id_{decision_measure._value_}.txt")
    old_ood_uncertainty = np.loadtxt(f"{org_eval_dir}/ood_{decision_measure._value_}.txt")
    old_uncertainty_pred = np.concatenate((old_id_uncertainty, old_ood_uncertainty), axis=0)
    if decision_measure == UncertaintyMeasuresEnum.CONFIDENCE:
        old_uncertainty_pred *= -1.0
        
    old_id_probs = np.loadtxt(f"{org_eval_dir}/id_probs.txt")
    old_ood_probs = np.loadtxt(f"{org_eval_dir}/ood_probs.txt")
    old_probs = np.concatenate((old_id_probs, old_ood_probs), axis=0)
    
    # new (under attack)
    target_epsilon_dir = os.path.join(attack_dir, f"e{epsilon}-attack")
    id_uncertainty = np.loadtxt(f"{target_epsilon_dir}/eval/{decision_measure._value_}.txt")
    ood_uncertainty = np.loadtxt(f"{target_epsilon_dir}/ood_eval/{decision_measure._value_}.txt")
    uncertainty_pred = np.concatenate((id_uncertainty, ood_uncertainty), axis=0)
    if decision_measure == UncertaintyMeasuresEnum.CONFIDENCE:
        uncertainty_pred *= -1.0
    id_probs = np.loadtxt(f"{target_epsilon_dir}/eval/probs.txt")
    ood_probs = np.loadtxt(f"{target_epsilon_dir}/ood_eval/probs.txt")
    probs = np.concatenate((id_probs, ood_probs), axis=0)
    
    id_labels = np.zeros_like(id_uncertainty)
    ood_labels = np.ones_like(ood_uncertainty)
    y_true = np.concatenate((id_labels, ood_labels), axis=0)
    
    # assign labels based on thershold given
    y_preds = np.zeros_like(uncertainty_pred)
    y_preds[uncertainty_pred >= threshold] = 1
    fp_indices = np.intersect1d(np.argwhere(y_true == 0), np.argwhere(y_preds == 1))
    fn_indices = np.intersect1d(np.argwhere(y_true == 1), np.argwhere(y_preds == 0))
    print(f"Cross check: fp: {len(fp_indices)}")
    print(f"Cross check: fn: {len(fn_indices)}")
    
    # create a separate dir to store all visualizations
    os.makedirs(os.path.join(target_epsilon_dir, plots_dir))
    
    # plot false positives
    fp_dir = os.path.join(target_epsilon_dir, plots_dir, 'in-out')
    os.makedirs(fp_dir)
    
    # first plot
    figure, axes = plt.subplots(nrows = 10, ncols=3, figsize=(15, 15))
    figure.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, index in enumerate(fp_indices):
        if index.ndim > 0:
            index = index[0]
        ri = i%10

        # org image
        axis = axes[ri][0]
        img = Image.open(f"{attack_dir}/org-images/{index}.png")
        axis.set_title(f"index: {index}, label: {int(y_true[index])}, {decision_measure._value_}: {np.round(old_uncertainty_pred[index],3)}", pad=2)
        #hide all spines
        axis.axis("off")
        axis.imshow(img, cmap="gray")

        # adv image
        axis = axes[ri][1]
        img = Image.open(f"{target_epsilon_dir}/adv-images/{index}.png")
        axis.set_title(f"label: {y_preds[index]}, {decision_measure._value_}: {np.round(uncertainty_pred[index],3)}", pad=2)
        #hide all spines
        axis.axis("off")
        axis.imshow(img, cmap="gray")

        # change in prob dist (bar plot)
        axis = axes[ri][2]
        class_labels = np.arange(10)
        width = 0.2
        rects1 = axis.bar(class_labels - width/2, old_probs[index, :], width, label='Org')
        rects2 = axis.bar(class_labels + width/2, probs[index, :], width, label='Adv')
        axis.set_xticks(class_labels)
        axis.set_xticklabels(class_labels)
        # places the legend to the right of the current axis
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # batch every 10 sample into a single image
        if (i > 0 and (i+1) % 10 == 0) or i == (fp_indices.size-1):
            plt.savefig(os.path.join(fp_dir, f"result_vis_{i+1}.png"), bbox_inches='tight')
            plt.close()

        if i > 0 and (i+1) % 10 == 0 and i != (fp_indices.size-1):
            figure, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 15))
            figure.subplots_adjust(hspace=0.5, wspace=0.5)

        if limit is not None and i == limit-1:
            plt.close()
            break

    # plot false negatives
    fn_dir = os.path.join(target_epsilon_dir, plots_dir, 'out-in')
    os.makedirs(fn_dir)
    
    # first plot
    figure, axes = plt.subplots(nrows = 10, ncols=3, figsize=(15, 15))
    figure.subplots_adjust(hspace=0.5, wspace=0.5)
    
    for i, index in enumerate(fn_indices):
        if index.ndim > 0:
            index = index[0]
        ri = i%10

        # org image
        axis = axes[ri][0]
        img = Image.open(f"{attack_dir}/org-images-ood/{index-len(id_uncertainty)}.png")
        axis.set_title(f"index: {index}, label: {int(y_true[index])}, {decision_measure._value_}: {np.round(old_uncertainty_pred[index],3)}", pad=2)
        #hide all spines
        axis.axis("off")
        axis.imshow(img, cmap="gray")

        # adv image
        axis = axes[ri][1]
        img = Image.open(f"{target_epsilon_dir}/adv-images-ood/{index-len(id_uncertainty)}.png")
        axis.set_title(f"label: {y_preds[index]}, {decision_measure._value_}: {np.round(uncertainty_pred[index],3)}", pad=2)
        #hide all spines
        axis.axis("off")
        axis.imshow(img, cmap="gray")

        # change in prob dist (bar plot)
        axis = axes[ri][2]
        class_labels = np.arange(10)
        width = 0.2
        rects1 = axis.bar(class_labels - width/2, old_probs[index, :], width, label='Org')
        rects2 = axis.bar(class_labels + width/2, probs[index, :], width, label='Adv')
        axis.set_xticks(class_labels)
        axis.set_xticklabels(class_labels)
        # places the legend to the right of the current axis
        axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # batch every 10 sample into a single image
        if (i > 0 and (i+1) % 10 == 0) or i == (fn_indices.size-1):
            plt.savefig(os.path.join(fn_dir, f"result_vis_{i+1}.png"), bbox_inches='tight')
            plt.close()

        if i > 0 and (i+1) % 10 == 0 and i != (fn_indices.size-1):
            figure, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 15))
            figure.subplots_adjust(hspace=0.5, wspace=0.5)

        if limit is not None and i == limit-1:
            plt.close()
            break

    return (len(fp_indices)+len(fn_indices))