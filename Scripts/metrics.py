import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def mean_absolute_difference(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_difference(y_true, y_pred):
    return np.mean(y_true - y_pred)

def std_difference(y_true, y_pred):
    return np.std(y_true - y_pred)

def percentage_MAD(y_true, y_pred):
    leq5 = 0
    leq10 = 0
    leq15 = 0

    for i in range(len(y_true)):
        if abs(y_true[i] - y_pred[i]) <= 5:
            leq5 += 1
        if abs(y_true[i] - y_pred[i]) <= 10:
            leq10 += 1
        if abs(y_true[i] - y_pred[i]) <= 15:
            leq15 += 1

    return leq5 * 100.0 / len(y_true), leq10 * 100.0 / len(y_true), leq15 * 100.0 / len(y_true)

def IEEE_STANDARD(MAD):
    if MAD <= 5:
        return 'A'
    elif MAD <= 6:
        return 'B'
    elif MAD <= 7:
        return 'C'
    else:
        return 'D'

def AAMI_STANDARD(SE, SDE):
    if abs(SE) <= 5 and SDE <= 8:
        return 'Pass'
    else:
        return 'Fail'

def BHS_STANDARD(leq5, leq10, leq15):
    if leq5 >= 60 and leq10 >= 85 and leq15 >= 95:
        return 'A'
    elif leq5 >= 50 and leq10 >= 75 and leq15 >= 90:
        return 'B'
    elif leq5 >= 40 and leq10 >= 65 and leq15 >= 85:
        return 'C'
    else:
        return 'D'

def evaluate_metrics(y_true, y_pred):
    sbp_MAD = mean_absolute_difference(y_true[:, 0], y_pred[:, 0])
    dbp_MAD = mean_absolute_difference(y_true[:, 1], y_pred[:, 1])
    sbp_MD = mean_difference(y_true[:, 0], y_pred[:, 0])
    dbp_MD = mean_difference(y_true[:, 1], y_pred[:, 1])
    sbp_SDE = std_difference(y_true[:, 0], y_pred[:, 0])
    dbp_SDE = std_difference(y_true[:, 1], y_pred[:, 1])
    sbp_leq5, sbp_leq10, sbp_leq15 = percentage_MAD(y_true[:, 0], y_pred[:, 0])
    dbp_leq5, dbp_leq10, dbp_leq15 = percentage_MAD(y_true[:, 1], y_pred[:, 1])

    sbp_IEEE = IEEE_STANDARD(sbp_MAD)
    dbp_IEEE = IEEE_STANDARD(dbp_MAD)
    sbp_AAMI = AAMI_STANDARD(sbp_MAD, sbp_SDE)
    dbp_AAMI = AAMI_STANDARD(dbp_MAD, dbp_SDE)
    sbp_BHS = BHS_STANDARD(sbp_leq5, sbp_leq10, sbp_leq15)
    dbp_BHS = BHS_STANDARD(dbp_leq5, dbp_leq10, dbp_leq15)

    # Convert all value into string with '%.2f' %
    sbp_MAD = '%.2f' % sbp_MAD
    dbp_MAD = '%.2f' % dbp_MAD
    sbp_MD = '%.2f' % sbp_MD
    dbp_MD = '%.2f' % dbp_MD
    sbp_SDE = '%.2f' % sbp_SDE
    dbp_SDE = '%.2f' % dbp_SDE
    sbp_leq5 = '%.1f' % sbp_leq5
    dbp_leq5 = '%.1f' % dbp_leq5
    sbp_leq10 = '%.1f' % sbp_leq10
    dbp_leq10 = '%.1f' % dbp_leq10
    sbp_leq15 = '%.1f' % sbp_leq15
    dbp_leq15 = '%.1f' % dbp_leq15


    # Create matplotlib table for the metrics
    aami_fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table = ax.table(cellText=[
        ['SBP', dbp_MD, dbp_SDE, dbp_AAMI],
        ['DBP', sbp_MD, sbp_SDE, sbp_AAMI],
    ],
        colLabels=['', 'ME', 'STD', 'AAMI Grade'],
        cellLoc='center',
        loc='center',
        colColours=['#ffadad', '#ffadad', '#ffadad', '#ffadad']
    )
    table.scale(1, 2)
    plt.subplots_adjust(0,0,1,1)
    plt.margins(0, 0)

    ieee_fig, ieee_fig_ax = plt.subplots(figsize=(4, 2))
    ieee_fig_ax.axis('off')
    table = ieee_fig_ax.table(cellText=[
        ['SBP', sbp_MAD, sbp_IEEE],
        ['DBP', dbp_MAD, dbp_IEEE]
    ],
        colLabels=['', 'MAD', 'IEEE Grade'],
        cellLoc='center',
        loc='center',
        colColours=['#ffd6a5', '#ffd6a5', '#ffd6a5']
    )
    table.scale(1, 2)
    plt.subplots_adjust(0,0,1,1)
    plt.margins(0, 0)

    bhs_fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    table = ax.table(cellText=[
        ['SBP', sbp_leq5, sbp_leq10, sbp_leq15, sbp_BHS],
        ['DBP', dbp_leq5, dbp_leq10, dbp_leq15, dbp_BHS]
    ],
        colLabels=['', '<= 5mmHg', '<= 10mmHg', '<= 15mmHg', 'BHS Grade'],
        cellLoc='center',
        loc='center',
        colColours=['#d9edf8', '#d9edf8', '#d9edf8', '#d9edf8', '#d9edf8']
    )
    table.scale(1, 2)
    plt.subplots_adjust(0,0,1,1)
    plt.margins(0, 0)

    # Create table of sample predictions and ground truth
    samples = []
    for _ in range(5):
        i = np.random.randint(len(y_true))
        samples.append([f'Sample {i}', f'{y_pred[i][0]:.2f}', f'{y_true[i][0]:.2f}', f'{y_pred[i][1]:.2f}', f'{y_true[i][1]:.2f}'])

    sample_fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    table = ax.table(cellText=samples,
        cellLoc='center',
        loc='center',
        colLabels=['', 'SBP Prediction', 'SBP Ground Truth', 'DBP Prediction', 'DBP Ground Truth'],
        colColours=['#dedaf4', '#dedaf4', '#dedaf4', '#dedaf4', '#dedaf4']
    )
    table.scale(1, 2)
    plt.subplots_adjust(0,0,1,1)
    plt.margins(0, 0)

    return ieee_fig, aami_fig, bhs_fig, sample_fig

def evaluate_classifier(y_true, y_pred):
    # Create confusion matrix
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], round(y_pred[i])] += 1

    # Calculate accuracy
    accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

    # Create matplotlib table for the metrics
    conf_fig, ax = plt.subplots(figsize=(6, 5))
    classes = ['Healthy', 'Unhealthy']
    sns.set(font_scale=1.2)
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues', cbar=True,
                xticklabels=classes, yticklabels=classes)

    # Add labels, title, and ticks
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix with Accuracy of {accuracy * 100:.2f}%')

    # Plotting ROC curve
    tpr_values, fpr_values = calculate_roc_curve(y_true, y_pred)
    roc_fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(fpr_values, tpr_values, color='b', lw=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    return conf_fig, roc_fig


def calculate_roc_curve(y_true, y_pred):
    # Sort predictions by descending order of scores
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_pred[sorted_indices]

    # Initialize lists to store TPR and FPR values
    tpr_values = []
    fpr_values = []

    # Calculate total number of positive and negative examples
    num_positive = np.sum(y_true == 1)
    num_negative = np.sum(y_true == 0)

    # Initialize counts of true positive (TP) and false positive (FP)
    tp_count = 0
    fp_count = 0

    # Iterate through sorted scores to compute TPR and FPR
    for i in range(len(y_scores_sorted)):
        if y_true_sorted[i] == 1:
            tp_count += 1
        else:
            fp_count += 1

        tpr = tp_count / num_positive
        fpr = fp_count / num_negative

        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return tpr_values, fpr_values

if __name__ == '__main__':
    y_true = np.random.rand(100, 1).round()
    y_pred = np.random.rand(100, 1).round()
    conf_fig, roc_fig = evaluate_classifier(y_true, y_pred)
    plt.show()

    # ieee_fig, aami_fig, bhs_fig, sample_fig = evaluate_metrics(y_true, y_pred)
    # plt.show()

"""
print('----------------------------------------------------------')
    print('|     | <= 5mmHg | <=10mmHg | <=15mmHg | BHS Grade |')
    print('----------------------------------------------------------')
    print('| DBP |  {:.1f} %  |  {:.1f} %  |  {:.1f} %  |    {}    '.format(sbp_leq5, sbp_leq10, sbp_leq15, sbp_BHS))
    print('| SBP |  {:.1f} %  |  {:.1f} %  |  {:.1f} %  |    {}    '.format(dbp_leq5, dbp_leq10, dbp_leq15, dbp_BHS))
    print('----------------------------------------------------------')

    print('----------------------------------------')
    print('|     |  ME   |  STD  | AAMI Grade |')
    print('----------------------------------------')
    print('| DBP | {:.2f} | {:.2f} |     {}     '.format(dbp_MD, dbp_SDE, dbp_AAMI))
    print('| SBP | {:.2f} | {:.2f} |     {}     '.format(sbp_MD, sbp_SDE, sbp_AAMI))
    print('----------------------------------------')

    print('----------------------------------------')
    print('|     |  MAD  | IEEE Grade |')
    print('----------------------------------------')
    print('| DBP | {:.2f} |     {}     '.format(dbp_MAD, dbp_IEEE))
    print('| SBP | {:.2f} |     {}     '.format(sbp_MAD, sbp_IEEE))
    print('----------------------------------------')
"""