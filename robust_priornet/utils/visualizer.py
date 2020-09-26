import os
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from PIL import Image
from ..eval.uncertainty import UncertaintyMeasuresEnum

def plot_curve(x, y, axis, x_label='', y_label='',
               title='', curve_title='',
               x_lim=None, y_lim=None,
               show_legend=False,
               axis_spine_visibility_config: list = None,
               title_wrap_length=20,
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

    axis.set_title("\n".join(wrap(title, title_wrap_length)))
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
    time_taken = []
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
        time_taken.append(summary['time_taken'])
    return train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy, train_loss, val_loss, time_taken

def get_step_summaries(model_dir, step_size):
    """
    Accumulates loss, accuracy values for train, val phases for every step summary
    written during training.
    """
    train_id_loss = []
    train_ood_loss = []
    val_id_loss = []
    val_ood_loss = []
    train_id_accuracy = []
    val_id_accuracy = []
    train_loss = []
    val_loss = []
    time_taken = []
    step = 0
    steps = []
    curr_path = os.path.join(model_dir, f'step_summary_{step+step_size}.pt')
    while os.path.exists(curr_path):
        summary = torch.load(curr_path)
        train_id_loss.append(summary['train_results']['id_loss'])
        train_ood_loss.append(summary['train_results']['ood_loss'])
        train_loss.append(summary['train_results']['loss'])
        train_id_accuracy.append(summary['train_results']['id_accuracy'])
        val_id_loss.append(summary['val_results']['id_loss'])
        val_ood_loss.append(summary['val_results']['ood_loss'])
        val_loss.append(summary['val_results']['loss'])
        val_id_accuracy.append(summary['val_results']['id_accuracy'])
        time_taken.append(summary['time_taken'])
        step += step_size
        steps.append(step)
        curr_path = os.path.join(model_dir, f'step_summary_{step+step_size}.pt')
    return steps, train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy, train_loss, val_loss, time_taken

def plot_loss_accuracy_curve(model_dir, num_epochs, step_size=None):
    if step_size is not None:
        x_values, train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy, train_loss, val_loss, _ = get_step_summaries(model_dir, step_size)
    else:
        train_id_loss, train_ood_loss, val_id_loss, val_ood_loss, train_id_accuracy, val_id_accuracy, train_loss, val_loss, _ = get_epoch_summaries(model_dir, num_epochs)

    figure, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
    x_axis_label = 'Epochs' if step_size is None else 'Steps'

    # ID axis for loss curve
    if step_size is None:
        x_values = np.arange(num_epochs)
    plot_curve(x_values, train_id_loss, axes[0,0], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='ID - Train loss', show_legend=True)
    plot_curve(x_values, val_id_loss, axes[0,0], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='ID - Validation loss', show_legend=True)

    # OOD axis for loss curve
    plot_curve(x_values, train_ood_loss, axes[0,1], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='OOD - Train loss', show_legend=True)
    plot_curve(x_values, val_ood_loss, axes[0,1], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='OOD - Validation loss', show_legend=True)

    # ID axis for loss curve (zoomed)
    if step_size is None:
        x_values = np.arange(num_epochs)
    plot_curve(x_values, train_id_loss, axes[1,0], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='ID - Train loss', show_legend=True, y_lim=[0, 5])
    plot_curve(x_values, val_id_loss, axes[1,0], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='ID - Validation loss', show_legend=True, y_lim=[0, 5])

    # OOD axis for loss curve (zoomed)
    plot_curve(x_values, train_ood_loss, axes[1,1], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='OOD - Train loss', show_legend=True, y_lim=[0, 0.1])
    plot_curve(x_values, val_ood_loss, axes[1,1], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='OOD - Validation loss', show_legend=True, y_lim=[0, 0.1])

    # ID axis for accuracy curve
    plot_curve(x_values, train_id_accuracy, axes[2,0], x_label=x_axis_label,
               y_label='Accuracy', title='Accuracy curve',
               curve_title='ID - Train accuracy', show_legend=True)
    plot_curve(x_values, val_id_accuracy, axes[2,0], x_label=x_axis_label,
               y_label='Accuracy', title='Accuracy curve',
               curve_title='ID - Validation accuracy', show_legend=True)

    # Overall loss axis
    plot_curve(x_values, train_loss, axes[2,1], x_label=x_axis_label,
               y_label='Loss', title='Loss curve',
               curve_title='Train loss', show_legend=True)
    plot_curve(x_values, val_loss, axes[2,1], x_label=x_axis_label,
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

# adv example plot - START
def plot_trio(save_path, model_dir, org_images_folder, attack_folders, img_index, indices, epsilon=0.2, is_ood=False, ood_dataset_name='', color_map='Greys'):
    """
    For the in-domain dataset, plots adv wrt diffE, distU and confidence (label misclassification)
    For the out-domain dataset, plots adv wrt diffE, distU.
    """
    color_dict, color_list, linestyle_dict, pgf_with_latex, default_params, label_dict = get_plot_params()
    sns.set(style='whitegrid', palette='colorblind', color_codes=True)
    if color_map == 'rgb':
        color_map = None
    plt.rcParams['lines.markersize'] = 0.2
    
    #plot results
    fig = plt.figure()
    cols = 3 if is_ood else 4
    widths = [2] * cols
    heights = [0.6, 0.6, 0.6]
    spec = fig.add_gridspec(ncols=cols, nrows=3, width_ratios=widths,
                              height_ratios=heights)
    # get org image
    axis_index = 0
    ax = fig.add_subplot(spec[0, axis_index])
    org_img = Image.open(f"{org_images_folder}/{img_index}.png")
    ax.axis("off")
    ax.set_title('Original')
    ax.imshow(org_img, cmap=color_map)
    axis_index+=1
    
    adv_images_folder = 'adv-images-ood' if is_ood else 'adv-images'
    # get confidence adv image (misclassify)
    if not is_ood:
        adv_indices = np.loadtxt(f"{attack_folders[0]}/e{epsilon}-attack/{adv_images_folder}/indices.txt")
        actual_index = np.argwhere(adv_indices == img_index)[0,0]
        print(f"Conf adv index: {actual_index}")
        ax = fig.add_subplot(spec[0, axis_index])
        conf_img = Image.open(f"{attack_folders[0]}/e{epsilon}-attack/{adv_images_folder}/{actual_index}.png")
        ax.axis("off")
        ax.set_title('$m_{conf}$ adv')
        ax.imshow(conf_img, cmap=color_map)
        axis_index += 1
    
    # get dE adv image (ood-detect)
    ax = fig.add_subplot(spec[0, axis_index])
    adv_indices = np.loadtxt(f"{attack_folders[1]}/e{epsilon}-attack/{adv_images_folder}/indices.txt")
    actual_index = np.argwhere(adv_indices == img_index)[0,0]
    print(f"DE adv index: {actual_index}")
    de_img = Image.open(f"{attack_folders[1]}/e{epsilon}-attack/{adv_images_folder}/{actual_index}.png")
    ax.axis("off")
    ax.set_title('$m_{diffE}$ adv')
    ax.imshow(de_img, cmap=color_map)
    axis_index += 1
    
    # get dU adv image (ood-detect)
    ax = fig.add_subplot(spec[0, axis_index])
    adv_indices = np.loadtxt(f"{attack_folders[2]}/e{epsilon}-attack/{adv_images_folder}/indices.txt")
    actual_index = np.argwhere(adv_indices == img_index)[0,0]
    print(f"DU adv index: {actual_index}")
    du_img = Image.open(f"{attack_folders[2]}/e{epsilon}-attack/{adv_images_folder}/{actual_index}.png")
    ax.axis("off")
    ax.set_title('$m_{distU}$ adv')
    ax.imshow(du_img, cmap=color_map)
    axis_index += 1
    
    # get precision adv image (ood-detect)
    #ax = fig.add_subplot(spec[0, 4])
    #adv_indices = np.loadtxt(f"{attack_folders[3]}/e{epsilon}-attack/{adv_images_folder}/indices.txt")
    #actual_index = np.argwhere(adv_indices == img_index)[0,0]
    #print(f"Precision adv index: {actual_index}")
    #pr_img = Image.open(f"{attack_folders[3]}/adv-images-ood/{actual_index}.png")
    #ax.axis("off")
    #ax.set_title('$m_{alpha_{0}}$ adv')
    #ax.imshow(pr_img, cmap=color_map)
    
    # alpha box plots
    eval_dir = 'ood_eval' if is_ood else 'eval'
    org_eval_dir = f'ood-eval{"-" if ood_dataset_name != "" else ""}{ood_dataset_name}' if is_ood else 'eval'
    org_logits_file = 'ood_logits' if is_ood else 'id_logits'
    org_alphas = np.exp(np.loadtxt(f"{model_dir}/{org_eval_dir}/{org_logits_file}.txt")[img_index])
    if not is_ood:
        conf_alphas = np.exp(np.loadtxt(f"{attack_folders[0]}/e{epsilon}-attack/{eval_dir}/logits.txt")[img_index])
    de_alphas = np.exp(np.loadtxt(f"{attack_folders[1]}/e{epsilon}-attack/{eval_dir}/logits.txt")[img_index])
    du_alphas = np.exp(np.loadtxt(f"{attack_folders[2]}/e{epsilon}-attack/{eval_dir}/logits.txt")[img_index])
    #pr_alphas = np.exp(np.loadtxt(f"{attack_folders[3]}/e{epsilon}-attack/{eval_dir}/logits.txt")[img_index])
    
    # conf box plot
    axis_index = 1
    if not is_ood:
        axis = fig.add_subplot(spec[1, axis_index])
        class_labels = np.arange(10)
        width = 0.4
        rects1 = axis.bar(class_labels - width/2, org_alphas, width, label='Org', color=color_list[7])
        rects2 = axis.bar(class_labels + width/2, conf_alphas, width, label='Adv', color=color_list[3])
        axis.set_xticks(class_labels)
        axis.set_xticklabels(class_labels)
        tick_spacing = max(np.max(org_alphas), np.max(conf_alphas))/5
        axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
        axis.set_title('Dirichlet parameters - $\\alpha$', fontsize=9)
        axis_index += 1
        # places the legend to the right of the current axis
        axis.legend(loc=0, fontsize=7)
    
    # DE box plot
    axis = fig.add_subplot(spec[1, axis_index])
    class_labels = np.arange(10)
    width = 0.4
    rects1 = axis.bar(class_labels - width/2, org_alphas, width, label='Org', color=color_list[7])
    rects2 = axis.bar(class_labels + width/2, de_alphas, width, label='Adv', color=color_list[3])
    #ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    axis.set_xticks(class_labels)
    axis.set_xticklabels(class_labels)
    axis.set_title('Dirichlet parameters - $\\alpha$', fontsize=9)
    tick_spacing = max(np.max(org_alphas), np.max(de_alphas))/4
    axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    axis_index += 1
    # places the legend to the right of the current axis
    axis.legend(loc=0, fontsize=7)
    axis.tick_params(axis='both', which='major', labelsize=9)
    
    # DU box plot
    axis = fig.add_subplot(spec[1, axis_index])
    class_labels = np.arange(10)
    width = 0.4
    rects1 = axis.bar(class_labels - width/2, org_alphas, width, label='Org', color=color_list[7])
    rects2 = axis.bar(class_labels + width/2, du_alphas, width, label='Adv', color=color_list[3])
    axis.set_xticks(class_labels)
    axis.set_xticklabels(class_labels)
    axis.set_title('Dirichlet parameters - $\\alpha$', fontsize=9)
    tick_spacing = max(np.max(org_alphas), np.max(du_alphas))/4
    axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    axis.tick_params(axis='both', which='major', labelsize=9)
    axis_index += 1
    # places the legend to the right of the current axis
    axis.legend(loc=0, fontsize=7)
    
    # Precision box plot
    #axis = fig.add_subplot(spec[1, 4])
    #class_labels = np.arange(10)
    #width = 0.4
    #rects1 = axis.bar(class_labels - width/2, org_alphas, width, label='Org')
    #rects2 = axis.bar(class_labels + width/2, pr_alphas, width, label='Adv')
    #axis.set_xticks(class_labels)
    #axis.set_xticklabels(class_labels)
    #axis.set_title('Dirichlet parameters', fontsize=7)
    # places the legend to the right of the current axis
    #axis.legend(loc=0)
    
    # confidence box plots
    eval_dir = 'ood_eval' if is_ood else 'eval'
    org_eval_dir = f'ood-eval{"-" if ood_dataset_name != "" else ""}{ood_dataset_name}' if is_ood else 'eval'
    org_probs_file = 'ood_probs' if is_ood else 'id_probs'
    org_probs = np.loadtxt(f"{model_dir}/{org_eval_dir}/{org_probs_file}.txt")[img_index]
    print(org_probs)
    if not is_ood:
        conf_probs = np.loadtxt(f"{attack_folders[0]}/e{epsilon}-attack/{eval_dir}/probs.txt")[img_index]
        print(conf_probs)
    de_probs = np.loadtxt(f"{attack_folders[1]}/e{epsilon}-attack/{eval_dir}/probs.txt")[img_index]
    print(de_probs)
    du_probs = np.loadtxt(f"{attack_folders[2]}/e{epsilon}-attack/{eval_dir}/probs.txt")[img_index]
    print(du_probs)
    #pr_probs = np.loadtxt(f"{attack_folders[3]}/e{epsilon}-attack/{eval_dir}/probs.txt")[img_index]
    #print(pr_probs)
    
    # conf box plot
    axis_index = 1
    if not is_ood:
        axis = fig.add_subplot(spec[2, axis_index])
        class_labels = np.arange(10)
        width = 0.4
        rects1 = axis.bar(class_labels - width/2, org_probs, width, label='Org', color=color_list[7])
        rects2 = axis.bar(class_labels + width/2, conf_probs, width, label='Adv', color=color_list[3])
        axis.set_xticks(class_labels)
        axis.set_xticklabels(class_labels)
        axis.set_title('Confidence in predictions - $(\\alpha/\\alpha_{0})$', fontsize=9)
        axis.tick_params(axis='both', which='major', labelsize=9)
        tick_spacing = max(np.max(org_probs), np.max(conf_probs))/4
        axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
        axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        axis_index += 1
        # places the legend to the right of the current axis
        axis.legend(loc=0, fontsize=7)
    
    # DE box plot
    axis = fig.add_subplot(spec[2, axis_index])
    class_labels = np.arange(10)
    width = 0.4
    rects1 = axis.bar(class_labels - width/2, org_probs, width, label='Org', color=color_list[7])
    rects2 = axis.bar(class_labels + width/2, de_probs, width, label='Adv', color=color_list[3])
    #ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    axis.set_xticks(class_labels)
    axis.set_xticklabels(class_labels)
    axis.set_title('Confidence in predictions - $(\\alpha/\\alpha_{0})$', fontsize=9)
    tick_spacing = max(np.max(org_probs), np.max(de_probs))/4
    axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    axis_index += 1
    # places the legend to the right of the current axis
    axis.legend(loc=0, fontsize=7)
    axis.tick_params(axis='both', which='major', labelsize=9)
    
    # DU box plot
    axis = fig.add_subplot(spec[2, axis_index])
    class_labels = np.arange(10)
    width = 0.4
    rects1 = axis.bar(class_labels - width/2, org_probs, width, label='Org', color=color_list[7])
    rects2 = axis.bar(class_labels + width/2, du_probs, width, label='Adv', color=color_list[3])
    axis.set_xticks(class_labels)
    axis.set_xticklabels(class_labels)
    axis.set_title('Confidence in predictions - $(\\alpha/\\alpha_{0})$', fontsize=9)
    tick_spacing = max(np.max(org_probs), np.max(du_probs))/4
    axis.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
    axis.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
    axis_index += 1
    # places the legend to the right of the current axis
    axis.legend(loc=0, fontsize=7)
    axis.tick_params(axis='both', which='major', labelsize=9)
    
    # Precision box plot
    #axis = fig.add_subplot(spec[2, 4])
    #class_labels = np.arange(10)
    #width = 0.4
    #rects1 = axis.bar(class_labels - width/2, org_probs, width, label='Org')
    #rects2 = axis.bar(class_labels + width/2, pr_probs, width, label='Adv')
    #axis.set_xticks(class_labels)
    #axis.set_xticklabels(class_labels)
    #axis.set_title('Confidence in predictions', fontsize=7)
    # places the legend to the right of the current axis
    #axis.legend(loc=0)
    
    save_path_curr = save_path.replace('.','_')
    savefig(save_path_curr, fig=fig)
    plt.close()
    
    mpl.rcParams.update(mpl.rcParamsDefault)

def get_common_indices(base_dir, attack_dirs, epsilon, is_ood=False):
    """
    Returns the indices where the images are adversaries in each of the attack
    folders given (intersection of adversaries wrt diffE, distU, confidence etc..)
    
    - uses the adv indices file stored in each attack dir.
    """
    adv_images_folder = 'adv-images-ood' if is_ood else 'adv-images'
    eval_folder = 'ood-eval' if is_ood else 'eval'
    common_indices = None
    for attack_dir in attack_dirs:
        misc = np.loadtxt(os.path.join(base_dir, attack_dir, f'e{epsilon}-attack', adv_images_folder, 'indices.txt'), dtype=np.int32)
        if common_indices is None:
            common_indices = misc
        else:
            common_indices = np.intersect1d(common_indices, misc)
    return common_indices
# adv example plot - END

def plot_epsilon_curve(epsilon: list, adv_success_rates: list,
                       result_dir: str = '.',
                       file_name: str = 'epsilon-curve.png',
                       plt_label: str = '',
                       title: str='',
                       save_fig=True,
                       plt_axis=None):
    if plt_axis is None:
        _, plt_axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    plt_axis.set_yticks(np.arange(0, 1.1, step=0.1))
    plt_axis.set_xticks(np.arange(0, np.max(epsilon)+0.1, step=0.1))
    plot_curve(epsilon, adv_success_rates, plt_axis, x_label='Epsilon',
               y_label='Adversarial Success Rate',
               x_lim=(0.0, np.max(epsilon)+ 0.1), y_lim=(0.0, 1.1),
               curve_title=plt_label, show_legend=(True if plt_label != '' else False),
               title=f'Adversarial Success Rate vs Epsilon - {title}',
               additional_plt_args=['*-'],
               title_wrap_length=100)
    if save_fig:
        plt.savefig(os.path.join(result_dir, file_name))

def plot_many_epsilon_curves(epsilon: list, adv_success_rates: list,
                             curve_legend_labels: list,
                             plot_title: str,
                             result_dir: str,
                             file_name: str):
    # send adv_success_rates as list of lists for each curve to be plotted.
    _, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    num_curves = len(adv_success_rates)
    for curve_index in range(num_curves):
        plot_epsilon_curve(epsilon, adv_success_rates[curve_index],
                           plt_label=curve_legend_labels[curve_index],
                           save_fig=False, plt_axis=axes, title=plot_title)
    plt.savefig(os.path.join(result_dir, file_name))
    plt.close()

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