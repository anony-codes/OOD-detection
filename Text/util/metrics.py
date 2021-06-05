
import numpy as np

from sklearn.metrics import roc_auc_score

def tpr95(in_confidence,out_confidence):
    # calculate the falsepositive error when tpr is 95%
    # calculate baseline
    T = 1
    # cifar-multiclass = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    # other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')


    start = 0.01
    end = 1
    gap = (end- start ) /100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = out_confidence
    X1 = in_confidence
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    fprBase = fpr /total

    # calculate our algorithm
    T = 1000
    # cifar-multiclass = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    # other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')

    start = 0.01
    end = 0.0104
    gap = (end- start ) /100000
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')

    Y1 = out_confidence
    X1 = in_confidence
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1

    fprNew = fpr /total

    return fprBase, fprNew



def auroc(in_df,ood_df,measure_type="argmax_prob"):

    argmax_prob_1_0 = np.array(list(in_df[in_df['gt'] == 0][measure_type])).reshape(-1)
    argmax_prob_1_1 = np.array(list(in_df[in_df['gt'] == 1][measure_type])).reshape(-1)


    argmax_prob_0_0 = np.array(list(ood_df[(ood_df['gt'] == 0)][measure_type])).reshape(-1)
    argmax_prob_0_1 = np.array(list(ood_df[(ood_df['gt'] == 1)][measure_type])).reshape(-1)


    argmax_prob_0 = np.concatenate((argmax_prob_0_0, argmax_prob_0_1), 0)
    argmax_prob_1 = np.concatenate((argmax_prob_1_0, argmax_prob_1_1), 0)

    label_1 = np.ones(len(argmax_prob_1))
    label_0 = np.zeros(len(argmax_prob_0))

    all_label = np.concatenate((label_0, label_1))
    all_max_prob = np.concatenate((argmax_prob_0, argmax_prob_1))

    auroc = roc_auc_score(all_label, all_max_prob)

    return auroc
import sklearn.metrics as skm

def get_roc_sklearn(xin, xood,criteria):

    xin = xin[criteria]
    xood = xood[criteria]

    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc