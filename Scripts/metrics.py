import numpy as np

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
    if SE <= 5 and SDE <= 8:
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

    print('----------------------------------------')
    print('|     | <= 5mmHg | <=10mmHg | <=15mmHg | BHS Grade |')
    print('----------------------------------------')
    print('| DBP |  {:.1f} %  |  {:.1f} %  |  {:.1f} %  |    {}    '.format(sbp_leq5, sbp_leq10, sbp_leq15, sbp_BHS))
    print('| SBP |  {:.1f} %  |  {:.1f} %  |  {:.1f} %  |    {}    '.format(dbp_leq5, dbp_leq10, dbp_leq15, dbp_BHS))
    print('----------------------------------------')

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


