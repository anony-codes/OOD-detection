import collections
import numpy as np


def kld(references, hypotheses, n_gram):

    r_cnter = count(references,n_gram)
    h_cnter = count(hypotheses,n_gram)

    s = set(r_cnter.keys())
    s.update(h_cnter.keys())
    s = list(s)
    r_probs = compute_probs(r_cnter, s)
    h_probs = compute_probs(h_cnter, s)
    kld = np.sum(r_probs * np.log(r_probs/h_probs))
    return kld


def count(x, n_gram):
    cnter = collections.Counter()
    for line in x:
        ngram_res = []
        temp = [-1] * (n_gram - 1) + line + [-1] * (n_gram - 1)
        for i in range(len(temp) + n_gram - 1):
            ngram_res.append(str(temp[i:i + n_gram]))
        cnter.update(ngram_res)
    return cnter

def compute_probs(cnter,token_lists):
    tot = 0
    probs = []
    for i in cnter:
        tot+= cnter[i]
    for i in token_lists:
        if i in cnter:
            probs.append(cnter[i] / tot)
        else:
            probs.append(1e-10)
    return np.array(probs)
