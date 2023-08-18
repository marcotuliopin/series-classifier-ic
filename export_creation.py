import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def plot_distribution(histogram, DATA_NAME, class_name, n, window_size, n_bins):
    """
    Plot histogram of the distribution of the data across all possible symbols present in it.
    """
    sns.set_theme(style='whitegrid')
    sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})  
    bin_count = int(np.ceil(np.log2(len(histogram))) + 1)
    f, ax = plt.subplots(figsize=(15, 8))
    fig = histogram.plot.hist(bins=bin_count)
    sns.set(font_scale=1.0)
    fig.bar_label(fig.containers[0])
    fig.set(title=f'Symbol\'s Appearence in {DATA_NAME} - class = {str(class_name)} - instance = {str(n)} - window size = {window_size} - n_bins = {n_bins}', ylabel='Appearence Count', xlabel='Symbol')
    # save figure
    Path(f"../fig/distribution/{DATA_NAME}/window_{window_size}/bins_{n_bins}/class_{str(class_name)}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../fig/distribution/{DATA_NAME}/window_{window_size}/bins_{n_bins}/class_{str(class_name)}/fig{str(n)}.png')
    plt.close()


def save_js_metrics(eq_class, diff_class, DATA_NAME, window_size, n_bins):
    """
    Save the results of the Jensenshannon distance calculation to a CSV and a TXT file.
    """
    Path(f'../metrics/{DATA_NAME}/window_{window_size}/bins_{n_bins}').mkdir(parents=True, exist_ok=True)

    # plot js difference
    plot_js_box(eq_class, diff_class, DATA_NAME, window_size, n_bins)

    with open(f'../metrics/{DATA_NAME}/window_{window_size}/bins_{n_bins}/js.txt', 'w') as file:
        same_class_mean = np.mean(eq_class)
        same_class_median = np.median(eq_class)
        diff_class_mean = np.mean(diff_class)
        diff_class_median = np.median(diff_class)
        difference = (1 - same_class_mean/diff_class_mean) * 100
        file.write('Same class JS - Mean: {}; Median: {}\n\n'.format(same_class_mean, same_class_median))
        file.write('Different classes JS - Mean: {}; Median: {}\n\n'.format(diff_class_mean, diff_class_median))
        file.write('Percentual difference between same class and different class: {}'.format(difference))


def plot_js_box(eq_class, diff_class, DATA_NAME, window_size, n_bins):
    """
    Plot boxplot of Jensen Shannon distance between samples of different classes and of the same class.
    """
    sns.set_theme(style='whitegrid')
    sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15})  
    plt.figure(figsize=(8, 6))

    combined_distances = pd.DataFrame({'Equal Class' : eq_class,
                                       'Different Class' : diff_class})

    fig = sns.boxplot(data=combined_distances, orient='h')
    fig.set_xlabel(f'Distance')
    Path(f'../fig/js/{DATA_NAME}/window_{window_size}/bins_{n_bins}').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../fig/js/{DATA_NAME}/window_{window_size}/bins_{n_bins}/fig.png')
    plt.close()


def save_classifier_metrics(acc_score, original_acc_score, cm, original_cm, DATA_NAME, window_size,
                             n_bins, y, label, algorithm):
    """
    Save the results of accuracy score and confusion matrix for the transformed and original data.
    """
    plot_confusion_matrix(original_cm, DATA_NAME, window_size, n_bins, label, {algorithm}, original=True)
    plot_confusion_matrix(cm, DATA_NAME, window_size, n_bins, y)
    with open(f'../metrics/{DATA_NAME}/window_{window_size}/bins_{n_bins}/{algorithm}/accuracy.txt', 'w') as file:
        file.write('With the transformation: {}\n\n'.format(acc_score))
        file.write('Without the transformation: {}\n\n'.format(original_acc_score))
        file.write('Difference in performance: {}'.format(acc_score - original_acc_score))


def plot_confusion_matrix(cm, DATA_NAME, window_size, n_bins, label, algorithm, original=False):
    val = np.mat(cm) 
    classnames = list(set(label))
    df_cm = pd.DataFrame(val, index=classnames, columns=classnames)
    plt.figure()
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Classifier Results')
    Path(f'../fig/conmat/{DATA_NAME}/window_{window_size}/bins_{n_bins}/{algorithm}').mkdir(parents=True, exist_ok=True)
    if original:
        plt.savefig(f'../fig/conmat/{DATA_NAME}/window_{window_size}/bins_{n_bins}/{algorithm}/og_fig.png')
    else:
        plt.savefig(f'../fig/conmat/{DATA_NAME}/window_{window_size}/bins_{n_bins}/{algorithm}/fig.png')
    plt.close()


def plot_accuracies(acc_score, og_acc_score, DATA_NAME, algorithm):
    plt.figure()
    plt.plot(acc_score, label='Modified')
    plt.plot(og_acc_score, label='Unmodified')
    plt.legend()
    Path(f'../fig/acc/{DATA_NAME}/{algorithm}').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../fig/acc/{DATA_NAME}/{algorithm}/fig.png')
    plt.close()


def plot_entropy_sc(comp_entrop, labels, DATA_NAME, window_size, n_bins):
    """
    Plot scatterplot of entropy x statistical complexity of the data.
    """
    sns.set_theme(style='whitegrid')
    sns.set_context("paper", rc={"font.size":15,"axes.titlesize":15,"axes.labelsize":15})  
    classes = np.unique(labels)

    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        indices = labels == classes[i]
        plt.scatter(comp_entrop.loc[indices, 'entropy'], comp_entrop.loc[indices, 'statistical_complexity'], label=classes[i], alpha=0.5, s=60)

    plt.xlabel('Entropy')
    plt.title(f'{DATA_NAME} - window size = {window_size} - n_bins = {n_bins}')
    plt.ylabel('Statistical Complexity')
    plt.legend()
    #save figure
    Path(f"../fig/entropy_sc/{DATA_NAME}/window_{window_size}/bins_{n_bins}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../fig/entropy_sc/{DATA_NAME}/window_{window_size}/bins_{n_bins}/fig.png')
    plt.close()