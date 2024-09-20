import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def mean_absolute_difference(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_difference(y_true, y_pred):
    return np.mean(y_true - y_pred)

def std_difference(y_true, y_pred):
    return np.std(y_true - y_pred, ddof=1)

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

def evaluate_metrics(y_pred, y_true):
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

def plot_histogram(y_preds, y_test):
    # Make histograms of the predictions and true values
    hist_fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(y_preds[:, 0], bins=50, alpha=0.5, label='SBP', color='#F8CECC', density=True)
    # ax[0].hist(y_test[:, 0], bins=50, alpha=0.5, label='SBP True', color='green', density = True)
    ax[0].legend()
    ax[0].set_title('SBP Histogram')
    ax[1].hist(y_preds[:, 1], bins=50, alpha=0.5, label='DBP', color='#DAE8FC', density=True)
    # ax[1].hist(y_test[:, 1], bins=50, alpha=0.5, label='DBP True', color='green', density = True)
    ax[1].legend()
    ax[1].set_title('DBP Histogram')
    return hist_fig

def plot_bland_altman(y_preds, y_test):
    # Create a Bland-Altman plot of SBP and DBP where x is mean and y is difference
    bland_altman_fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    sbp_diff = y_preds[:, 0] - y_test[:, 0]
    dbp_diff = y_preds[:, 1] - y_test[:, 1]
    sbp_mean = (y_preds[:, 0] + y_test[:, 0]) / 2
    dbp_mean = (y_preds[:, 1] + y_test[:, 1]) / 2
    ax[0].scatter(sbp_mean, sbp_diff, color='#F8CECC', s=20)
    ax[0].set_title('SBP Bland-Altman Plot')
    ax[1].scatter(dbp_mean, dbp_diff, color='#DAE8FC', s=20)
    ax[1].set_title('DBP Bland-Altman Plot')
    # Add mean line and 95% limits of agreement
    sbp_mean_diff = np.mean(sbp_diff)
    sbp_std_diff = np.std(sbp_diff)
    dbp_mean_diff = np.mean(dbp_diff)
    dbp_std_diff = np.std(dbp_diff)
    sbp_upper_limit = sbp_mean_diff + 1.96 * sbp_std_diff
    sbp_lower_limit = sbp_mean_diff - 1.96 * sbp_std_diff
    dbp_upper_limit = dbp_mean_diff + 1.96 * dbp_std_diff
    dbp_lower_limit = dbp_mean_diff - 1.96 * dbp_std_diff

    # Add mean line and 95% limits of agreement with labels
    ax[0].axhline(sbp_mean_diff, color='red', linestyle='-',
                  label=f'Mean difference ({sbp_mean_diff:.2f})')
    ax[0].axhline(sbp_upper_limit, color='black', linestyle='--',
                  label=f'Upper 95% LoA ({sbp_upper_limit:.2f})')
    ax[0].axhline(sbp_lower_limit, color='black', linestyle='--',
                  label=f'Lower 95% LoA ({sbp_lower_limit:.2f})')

    ax[1].axhline(dbp_mean_diff, color='blue', linestyle='-',
                  label=f'Mean difference ({dbp_mean_diff:.2f})')
    ax[1].axhline(dbp_upper_limit, color='black', linestyle='--',
                  label=f'Upper 95% LoA ({dbp_upper_limit:.2f})')
    ax[1].axhline(dbp_lower_limit, color='black', linestyle='--',
                  label=f'Lower 95% LoA ({dbp_lower_limit:.2f})')
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[0].set_xlabel('Mean of SBP (mmHg)')
    ax[0].set_ylabel('Difference of SBP (mmHg)')
    ax[1].set_xlabel('Mean of DBP (mmHg)')
    ax[1].set_ylabel('Difference of DBP (mmHg)')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    return bland_altman_fig

def plot_error_histogram(y_preds, y_test):
    # Make histogram of error
    error = y_preds - y_test
    hist_error_fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].hist(error[:, 0], bins=50, alpha=0.5, label='SBP Error', color='red')
    ax[0].legend()
    ax[0].set_title('SBP Error Histogram')
    ax[1].hist(error[:, 1], bins=50, alpha=0.5, label='DBP Error', color='blue')
    ax[1].legend()
    ax[1].set_title('DBP Error Histogram')
    return hist_error_fig

def plot_top_5_error_signals(y_preds, y_test, test_dataset):
    # Get the top 5 errors
    error = y_preds - y_test
    top_10_error = np.argsort(np.abs(error.sum(axis=1)))[-5:]
    # Draw the ppg signals
    ppg_error_fig, ax = plt.subplots(5, 1, figsize=(10, 20))
    for i, idx in enumerate(top_10_error):
        ppg = test_dataset.inputs[idx]['signals'][0]
        ax[i].plot(ppg)
        ax[i].set_title(f'PPG Signal {idx} with error {error[idx]}')
    plt.tight_layout()

if __name__ == '__main__':
    y_true = np.random.rand(100, 2)
    y_pred = np.random.rand(100, 2)
    ieee_fig, aami_fig, bhs_fig, sample_fig = evaluate_metrics(y_true, y_pred)
    plt.show()

