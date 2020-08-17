#!/usr/bin/env python
# coding: utf-8

import xnet
import glob
import json
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from igraph import *
import scipy
from scipy.stats import pearsonr
from collections import defaultdict
from util import save, load

from numpy.random import RandomState
# random_state = RandomState(seed=10)
# print(random_state.get_state()[1][0])
from datetime import datetime

import warnings

warnings.filterwarnings("error")


def get_temporal_series(valid_authors, data, min_year):
    Y = []
    for author in valid_authors:
        history = data[author]
        y = []
        old_year = 0
        for year in range(1995, 2011):
            try:
                value = history[str(year)]
            except:
                value = 0
            if year >= min_year:
                y.append(value)
        Y.append(np.asarray(y))
    Y = np.asarray(Y)
    return Y


'''
autores válidos seguindo critério frouxo
'''


def read_valid_authors():
    import glob
    name_to_authors = dict()
    # files = glob.glob('data2/valid_authors_min_criteria_*.txt')
    files = ['data2/valid_authors_min_criteria_in_out_10_10.txt']
    for valid_authors_file in files:
        valid_authors = open(valid_authors_file, 'r').read().split("\n")[:-1]
        name_to_authors[valid_authors_file[:-4]] = valid_authors
    return name_to_authors


def get_authors_by_percentile(author_values, key_header):
    authors = list(author_values.keys())
    values = list(author_values.values())
    authors = np.asarray(authors)
    values_space = np.percentile(values, [0, 25, 50, 75, 100])
    values_space[-1] += 1
    print(values_space)

    author2class = np.searchsorted(values_space, values, 'right')

    authors_by_class = dict()
    unique_values = np.unique(author2class)
    for c in unique_values:
        authors_by_class[key_header + str(c)] = authors[author2class == c]
    return authors_by_class


def delta_temporal_series(X, delta):
    Y = []
    for x in X:
        y = []
        for i in range(len(x) - delta):
            y.append(x[i + delta] - x[i])
        Y.append(np.asarray(y))
    Y = np.asarray(Y)
    return Y


def null_model(data, bins=np.linspace(-1, 1, 32), iters=1000):
    X, Y = data
    try:
        true_corr = pearsonr(X, Y)[0]
    except:
        print(X, Y)

    n = len(X)
    idxs1 = np.arange(n)
    idxs2 = np.arange(n)
    corrs = []

    dt = datetime.now()

    random_state = RandomState(seed=dt.microsecond)
    for i in range(iters):
        random_state.shuffle(idxs1)
        random_state.shuffle(idxs2)
        corr = pearsonr(X[idxs1], Y[idxs2])[0]
        corrs.append(corr)
    corrs = np.asarray(corrs)
    nonzero = np.count_nonzero(np.abs(corrs) > np.abs(true_corr))
    p_val = nonzero / iters

    hist, _ = np.histogram(corrs, bins=bins)

    return np.asarray([true_corr, p_val, hist], dtype=object)


def corr_temporal_series_col_null(pool, random_state, temporal_x, temporal_y, title, file):
    n_bins = 32
    bins = np.linspace(-1, 1, n_bins)

    corrs = []
    for x, y in zip(temporal_x, temporal_y):
        c = pearsonr(x, y)[0]
        corrs.append(c)
    corrs = np.asarray(corrs).reshape((-1, 1))

    iters = 1000
    count = np.zeros((len(temporal_x), iters))
    idxs = np.arange(len(temporal_x))
    for k in range(iters):
        temp_x = np.zeros(temporal_x.shape)
        temp_y = np.zeros(temporal_y.shape)
        for i in range(16):
            random_state.shuffle(idxs)
            temp_x[:, i] = temporal_x[idxs, i]
            random_state.shuffle(idxs)
            temp_y[:, i] = temporal_y[idxs, i]
        p = pool.starmap(pearsonr, list(zip(temp_x, temp_y)))
        p = np.asarray(p)[:, 0]
        count[:, k] = p

    p_vals = np.count_nonzero(count < corrs, axis=1) / iters
    hists = [np.histogram(c, bins=bins)[0] for c in count]
    hists = np.asarray(hists)
    hist_ave = hists.mean(0)
    hist_std = hists.std(0)

    corrs = corrs.flatten()
    idxs_le = p_vals > 0.05
    corr_le = corrs[idxs_le]
    label_le = 'p-value > 0.05'

    idxs_g_pos = np.logical_and(p_vals <= 0.05, corrs >= 0)
    corr_g_pos = corrs[idxs_g_pos]
    label_g_pos = 'p-value <= 0.05 (corr >= 0)'

    idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
    corr_g_neg = corrs[idxs_g_neg]
    label_g_neg = 'p-value <= 0.05 (corr < 0)'

    # HIST STACKED
    plt.title(title)
    plt.xlim(-1, 1)
    plt.hist([corr_g_pos, corr_g_neg, corr_le],
             bins=bins, alpha=0.6, stacked=True,
             color=['orange', 'blue', 'gray'],
             label=[label_g_pos, label_g_neg, label_le])
    plt.legend(loc="upper right")

    # CORR INFOS MEAN AND STD
    mu = np.nanmean(corrs)
    sigma = np.nanstd(corrs)
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    props = dict(boxstyle='round', facecolor='gray', alpha=0.3)
    ax = plt.gca()
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)

    ax.text(0.45, 0.1, "%.2f%%" % (100 * len(corr_le) / len(corrs)),
            color='gray', transform=ax.transAxes, fontsize=12)
    ax.text(0.85, 0.1, "%.2f%%" % (100 * len(corr_g_pos) / len(corrs)),
            color='orange', transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.1, "%.2f%%" % (100 * len(corr_g_neg) / len(corrs)),
            color='blue', transform=ax.transAxes, fontsize=12)

    # NULL MODEL PLOT1

    x_bin_center = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins - 1)]
    plt.errorbar(x_bin_center, hist_ave, yerr=hist_std, color='red')
    plt.savefig('temporal_series_data/' + file + '_col_null.pdf')
    plt.clf()
    return corrs


def corr_temporal_series_curves(pool, files, X, Y, title, filename):
    fig, axes = plt.subplots(4, 2, figsize=(8, 9), sharex=True)
    i_axes = 0
    for file in files:
        temporal_x = X[file]
        temporal_y = Y[file]
        # output = pool.map(null_model, list(zip(temporal_x, temporal_y)))
        # output = np.asarray(output)
        # np.savetxt(filename+file+'_data.csv', output[:,:2], delimiter=',')
        output = np.loadtxt(filename + file + '_data.csv', delimiter=',')

        corrs = output[:, 0]
        p_vals = output[:, 1]

        idxs_le = p_vals > 0.05
        corr_le = corrs[idxs_le]
        label_le = 'p-value > 0.05'

        idxs_g_pos = np.logical_and(p_vals <= 0.05, corrs >= 0)
        corr_g_pos = corrs[idxs_g_pos]
        label_g_pos = 'p-value <= 0.05 (corr >= 0)'

        idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
        corr_g_neg = corrs[idxs_g_neg]
        label_g_neg = 'p-value <= 0.05 (corr < 0)'

        # CURVES
        corr_le_curves_x = temporal_x[idxs_le]
        corr_le_curves_y = temporal_y[idxs_le]

        # for

        corr_g_pos_curves_x = temporal_x[idxs_g_pos]
        corr_g_pos_curves_y = temporal_y[idxs_g_pos]

        corr_g_neg_curves_x = temporal_x[idxs_g_neg]
        corr_g_neg_curves_y = temporal_y[idxs_g_neg]

        axes[i_axes % 4][i_axes // 4].errorbar(np.arange(1995, 2011), corr_le_curves_x.mean(0),
                                               yerr=corr_le_curves_x.std(0) / 2, c='gray', alpha=0.5,
                                               label='x ' + label_le, ls='-')
        axes[i_axes % 4][i_axes // 4].errorbar(np.arange(1995, 2011), corr_le_curves_y.mean(0),
                                               yerr=corr_le_curves_y.std(0) / 2, c='gray', alpha=0.5,
                                               label='y ' + label_le, ls='dashed')

        axes[i_axes % 4][i_axes // 4].errorbar(np.arange(1995, 2011), corr_g_pos_curves_x.mean(0),
                                               yerr=corr_g_pos_curves_x.std(0) / 2, c='orange', alpha=0.5,
                                               label='x ' + label_g_pos, ls='-')
        axes[i_axes % 4][i_axes // 4].errorbar(np.arange(1995, 2011), corr_g_pos_curves_y.mean(0),
                                               yerr=corr_g_pos_curves_y.std(0) / 2, c='orange', alpha=0.5,
                                               label='y ' + label_g_pos, ls='dashed')

        axes[i_axes % 4][i_axes // 4].errorbar(np.arange(1995, 2011), corr_g_neg_curves_x.mean(0),
                                               yerr=corr_g_neg_curves_x.std(0) / 2, c='blue', alpha=0.5,
                                               label='x ' + label_g_neg, ls='-')
        axes[i_axes % 4][i_axes // 4].errorbar(np.arange(1995, 2011), corr_g_neg_curves_y.mean(0),
                                               yerr=corr_g_neg_curves_y.std(0) / 2, c='blue', alpha=0.5,
                                               label='y ' + label_g_neg, ls='dashed')

        axes[i_axes % 4][i_axes // 4].set_title(file)
        axes[i_axes % 4][i_axes // 4].legend(prop={'size': 5})

        i_axes += 1
    # plt.legend(bbox_to_anchor = (0.9, 1))
    fig.savefig('temporal_series_data/' + filename + '_curves.pdf')
    plt.clf()


def corr_temporal_series_curves_cit_ref(pool, files, X, Y, title, filename):
    i_axes = 0
    for file in files:
        temporal_x = X[file]
        temporal_y = Y[file]
        # output = pool.map(null_model, list(zip(temporal_x, temporal_y)))
        # output = np.asarray(output)
        output = np.loadtxt(filename + file + '_data.csv', delimiter=',')
        corrs = output[:, 0]
        p_vals = output[:, 1]

        idxs_le = p_vals > 0.05
        corr_le = corrs[idxs_le]
        label_le = 'p-value > 0.05'

        idxs_g_pos = np.logical_and(p_vals <= 0.05, corrs >= 0)
        corr_g_pos = corrs[idxs_g_pos]
        label_g_pos = 'p-value <= 0.05 (corr >= 0)'

        idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
        corr_g_neg = corrs[idxs_g_neg]
        label_g_neg = 'p-value <= 0.05 (corr < 0)'

        # CURVES
        corr_le_curves_x = temporal_x[idxs_le]
        corr_le_curves_y = temporal_y[idxs_le]

        # for

        corr_g_pos_curves_x = temporal_x[idxs_g_pos]
        corr_g_pos_curves_y = temporal_y[idxs_g_pos]

        corr_g_neg_curves_x = temporal_x[idxs_g_neg]
        corr_g_neg_curves_y = temporal_y[idxs_g_neg]

        alphas = np.linspace(0.3, 1, 5)

        x = np.arange(len(corr_le_curves_x))
        idxs = np.abs(corr_le) > 0.6
        x = x[idxs]
        random_state.shuffle(x)
        j = 0
        for i in x[:5]:
            plt.plot(corr_le_curves_x[i], corr_le_curves_y[i], 'o-', c='gray', alpha=alphas[j], label=label_le)
            j += 1

        x = np.arange(len(corr_g_pos_curves_x))
        idxs = np.abs(corr_g_pos) > 0.8
        x = x[idxs]
        random_state.shuffle(x)
        j = 0
        for i in x[:5]:
            plt.plot(corr_g_pos_curves_x[i], corr_g_pos_curves_y[i], 'o-', c='orange', alpha=alphas[j],
                     label=label_g_pos)
            j += 1

        x = np.arange(len(corr_g_neg_curves_x))
        idxs = np.abs(corr_g_neg) > 0.8
        x = x[idxs]
        random_state.shuffle(x)
        j = 0
        for i in x[:5]:
            plt.plot(corr_g_neg_curves_x[i], corr_g_neg_curves_y[i], 'o-', c='blue', alpha=alphas[j], label=label_g_neg)
            j += 1

        plt.xlabel('refs')
        plt.ylabel('cits')
        plt.title(title)
        plt.savefig('temporal_series_data/' + filename + file + '_refs_cits_groups.pdf')
        plt.clf()
        i_axes += 1


def corr_temporal_series_curves_samples(pool, files, X, Y, title, filename):
    i_axes = 0
    for file in files:
        temporal_x = X[file]
        temporal_y = Y[file]
        try:
            output = np.loadtxt(filename + file + '_data.csv', delimiter=',')
        except:
            output = pool.map(null_model, list(zip(temporal_x, temporal_y)))
            output = np.asarray(output)

        corrs = output[:, 0]
        p_vals = output[:, 1]

        idxs_le = p_vals > 0.05
        corr_le = corrs[idxs_le]
        label_le = 'p-value > 0.05'

        idxs_g_pos = np.logical_and(p_vals <= 0.05, corrs >= 0)
        corr_g_pos = corrs[idxs_g_pos]
        label_g_pos = 'p-value <= 0.05 (corr >= 0)'

        idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
        corr_g_neg = corrs[idxs_g_neg]
        label_g_neg = 'p-value <= 0.05 (corr < 0)'

        # CURVES
        corr_le_curves_x = temporal_x[idxs_le]
        corr_le_curves_y = temporal_y[idxs_le]

        # for

        corr_g_pos_curves_x = temporal_x[idxs_g_pos]
        corr_g_pos_curves_y = temporal_y[idxs_g_pos]

        corr_g_neg_curves_x = temporal_x[idxs_g_neg]
        corr_g_neg_curves_y = temporal_y[idxs_g_neg]

        fig, axes = plt.subplots(2, 1, sharex=True)
        time = np.arange(1995, 2011)
        for x in corr_le_curves_x:
            axes[0].plot(time, x, alpha=0.2, c='gray')
        for y in corr_le_curves_y:
            axes[1].plot(time, y, alpha=0.2, c='gray')
        axes[0].set_title('x')
        axes[1].set_title('y')
        fig.savefig('temporal_series_data/' + filename + file + 'corr_le_samples.pdf')
        fig.clf()

        fig, axes = plt.subplots(2, 1, sharex=True)
        time = np.arange(1995, 2011)
        for x in corr_g_pos_curves_x:
            axes[0].plot(time, x, alpha=0.2, c='orange')
        for y in corr_g_pos_curves_y:
            axes[1].plot(time, y, alpha=0.2, c='orange')
        axes[0].set_title('x')
        axes[1].set_title('y')
        fig.savefig('temporal_series_data/' + filename + file + 'corr_g_pos_samples.pdf')
        fig.clf()

        fig, axes = plt.subplots(2, 1, sharex=True)
        time = np.arange(1995, 2011)
        for x in corr_g_neg_curves_x:
            axes[0].plot(time, x, alpha=0.2, c='blue')
        for y in corr_g_neg_curves_y:
            axes[1].plot(time, y, alpha=0.2, c='blue')
        axes[0].set_title('x')
        axes[1].set_title('y')
        fig.savefig('temporal_series_data/' + filename + file + 'corr_g_neg_samples.pdf')
        fig.clf()


def corr_temporal_series(pool, temporal_x, temporal_y, title, file):
    n_bins = 32
    bins = np.linspace(-1, 1, n_bins)

    output = pool.map(null_model, list(zip(temporal_x, temporal_y)))
    output = np.asarray(output)
    corrs = output[:, 0]
    p_vals = output[:, 1]
    hist_ave = output[:, 2].mean(0)
    hist_std = output[:, 2].std(0)

    '''
    for x,y in zip(temporal_x,temporal_y):
        corr = pearsonr(x,y)[0]
        corr,p_val,hist = null_model(x,y,bins,iters)
        
        corrs.append(corr)
        p_vals.append(p_val)
        hist_ave += hist
    '''

    idxs_le = p_vals > 0.05
    corr_le = corrs[idxs_le]
    label_le = 'p-value > 0.05'

    idxs_g_pos = np.logical_and(p_vals <= 0.05, corrs >= 0)
    corr_g_pos = corrs[idxs_g_pos]
    label_g_pos = 'p-value <= 0.05 (corr >= 0)'

    idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
    corr_g_neg = corrs[idxs_g_neg]
    label_g_neg = 'p-value <= 0.05 (corr < 0)'

    # HIST STACKED
    plt.title(title)
    plt.xlim(-1, 1)
    plt.hist([corr_g_pos, corr_g_neg, corr_le],
             bins=bins, alpha=0.6, stacked=True,
             color=['orange', 'blue', 'gray'],
             label=[label_g_pos, label_g_neg, label_le])
    plt.legend(loc="upper right")

    # CORR INFOS MEAN AND STD
    mu = np.nanmean(corrs)
    sigma = np.nanstd(corrs)
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,)))
    props = dict(boxstyle='round', facecolor='gray', alpha=0.3)
    ax = plt.gca()
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', bbox=props)

    ax.text(0.45, 0.1, "%.2f%%" % (100 * len(corr_le) / len(corrs)),
            color='gray', transform=ax.transAxes, fontsize=12)
    ax.text(0.85, 0.1, "%.2f%%" % (100 * len(corr_g_pos) / len(corrs)),
            color='orange', transform=ax.transAxes, fontsize=12)
    ax.text(0.05, 0.1, "%.2f%%" % (100 * len(corr_g_neg) / len(corrs)),
            color='blue', transform=ax.transAxes, fontsize=12)

    # NULL MODEL PLOT1

    x_bin_center = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins - 1)]
    plt.errorbar(x_bin_center, hist_ave, yerr=hist_std, color='red')
    plt.tight_layout()
    plt.savefig('temporal_series_data/' + file + '.pdf')
    plt.clf()
    return corrs


def filter_const(X, Y):
    X2 = []
    Y2 = []
    for x, y in zip(X, Y):
        if np.count_nonzero(y == y[0]) == len(y) or np.count_nonzero(x == x[0]) == len(x):
            continue

        X2.append(x)
        Y2.append(y)
    X2 = np.asarray(X2)
    Y2 = np.asarray(Y2)
    return X2, Y2


'''
    paper_count = defaultdict(lambda: 0)
    paper_cit_count = defaultdict(lambda: 0)

    print(data.vcount())
    i = 0
    for key, valid_authors in files_valid_authors.items():
        for paper in data.vs:
            if paper['year'] < 1995 or paper['year'] > 2010:
                continue
            i += 1
            authors_idxs = paper['authors_idxs'].split(',')
            for author in authors_idxs:
                if author in valid_authors:
                    paper_count[author] += 1
                    paper_cit_count[author] += len(paper.neighbors(mode=IN))
            if i % 100000 == 0:
                print(i)
        save(paper_count, 'authors_paper_count.json')
        save(paper_cit_count, 'authors_cit_count.json')
        break
    '''


def plot_history(files_valid_authors, data, valid_key, filename):
    authors_hist = defaultdict(lambda: np.arange(16))
    for paper in data.vs:
        year = int(paper['year'])
        authors = paper['authors_idxs'].split(',')
        for author in authors:
            authors_hist[author][year - 1995] += 1

    x = np.arange(1995, 2011)
    for key, valid_authors in files_valid_authors.items():
        hists = []
        if valid_key in key:
            for author, hist in authors_hist.items():
                if author in valid_authors:
                    hists.append(hist)

            hists = np.asarray(hists)
            y = hists.mean(0)
            y_error = hists.std(0) / 2
            plt.errorbar(x, y, label=key, yerr=y_error, alpha=0.7)
        plt.title('Número médio de artigos por ano')
        plt.legend()
        plt.savefig(filename)
    plt.clf()


#
# from = 5
# to = 3
# 
# [_from_] [_to_]
# 
# from:
# citacoes de papers de [_from_]  em [_from_]
# div in/out de papers de [_from_] em [_from_]
# 
# to:
# citFrom: citacoes de papers de [_from_]   em [_to_]
# div inFrom: div in de papers de [_from_]  em [_to_] 
# 
# citAll: citacoes de papers de [_from_]   em  [_from_][_to_]
# div inAll: div in de papers de [_from_]  em [_from_][_to_]
# 
# div out: div out de papers de [_to_]  em [_to_]
# 

if __name__ == '__main__':

    # ESQUERDO -> DIREITO

    # dados gerados para os autores considerando o intervalo do passado de 5 anos
    # ESSES SÃO OS REGISTROS DO LADO ESQUERTO
    authors_out_per_paper = load('data2/authors_out_freq_by_paper.json')
    authors_in_per_paper = load('data2/authors_in_freq_by_paper.json')  # citações
    authors_out = load('data2/authors_out_freq.json')
    authors_in = load('data2/authors_in_freq.json')  # citações

    # ESSES SÃO OS REGISTROS DO LADO DIREITO
    cit_from_citations = load('data2/cit_from_citations_per_paper.json')
    cit_from_diversity = load('data2/cit_from_diversity.json')
    cit_all_citations = load('data2/cit_all_citations_per_paper.json')
    cit_all_diversity = load('data2/cit_all_diversity.json')
    out_to_to_citations = load('data2/out_to_to_citations_per_paper.json')
    out_to_to_diversity = load('data2/out_to_to_diversity.json')

    authors_in_div = load('data2/authors_in_div.json')
    authors_out_div = load('data2/authors_out_div.json')

    files_valid_authors = read_valid_authors()
    print(files_valid_authors.keys())

    data = xnet.xnet2igraph('data/citation_network_ge1985_pacs.xnet')

    paper_count = load('authors_paper_count.json')
    paper_cit_count = load('authors_cit_count.json')

    authors_by_paper_count = get_authors_by_percentile(paper_count, 'paper_')
    authors_by_paper_cit = get_authors_by_percentile(paper_cit_count, 'cit_')
    files_valid_authors = {**authors_by_paper_count, **authors_by_paper_cit}

    files = list(files_valid_authors.keys())
    temporal_series_div_out = dict()
    temporal_series_div_in = dict()
    temporal_series_in = dict()
    temporal_series_out = dict()

    future_from_to_cit = dict()
    future_from_to_div = dict()
    future_out_to_to_ref = dict()
    future_out_to_to_div = dict()
    for file, valid_authors in files_valid_authors.items():
        temporal_series_div_out[file] = get_temporal_series(valid_authors, authors_out_div, 1995)
        temporal_series_div_in[file] = get_temporal_series(valid_authors, authors_in_div, 1995)
        temporal_series_in[file] = get_temporal_series(valid_authors, authors_in_per_paper, 1995)
        temporal_series_out[file] = get_temporal_series(valid_authors, authors_out_per_paper, 1995)

        future_from_to_cit[file] = get_temporal_series(valid_authors, cit_from_citations, 1995)
        future_from_to_div[file] = get_temporal_series(valid_authors, cit_from_diversity, 1995)
        future_out_to_to_ref[file] = get_temporal_series(valid_authors, out_to_to_citations, 1995)
        future_out_to_to_div[file] = get_temporal_series(valid_authors, out_to_to_diversity, 1995)

    # plot_history(files_valid_authors, data, 'paper', 'authors_paper_count_history.pdf')
    # plot_history(files_valid_authors, data, 'cit', 'authors_cit_history.pdf')

    pool = multiprocessing.Pool(processes=6)

    # corr_temporal_series_curves_samples(pool, files, temporal_series_out, future_from_to_cit,
    #                             'Correlation between refs(PAST->PAST) and cit(FUT->PAST)',
    #                             'corr_ref_past_cit_fut_')
    #
    # corr_temporal_series_curves_samples(pool, files, temporal_series_div_out, future_from_to_cit,
    #                             'Correlation between div out(PAST->PAST) and cit(FUT->PAST)',
    #                             'corr_div_out_past_cit_fut_')
    #
    # corr_temporal_series_curves(pool, files, temporal_series_out, future_from_to_cit,
    #                             'Correlation between refs(PAST->PAST) and cit(FUT->PAST)',
    #                             'corr_ref_past_cit_fut_')
    # corr_temporal_series_curves_cit_ref(pool, files, temporal_series_out, future_from_to_cit,
    #                             'Correlation between refs(PAST->PAST) and cit(FUT->PAST)',
    #                             'corr_ref_past_cit_fut_')

    for file in files:
        print(file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_out[file], future_out_to_to_ref[file]),
                             'Correlation between div out(PAST->PAST) and refs(FUT->FUT)',
                             'corr_div_out_past_ref_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_out[file], future_out_to_to_div[file]),
                            'Correlation between div out(PAST->PAST) and div out(FUT->FUT)',
                            'corr_div_out_past_div_out_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_in[file], future_out_to_to_div[file]),
                            'Correlation between cit(PAST->PAST) and div out(FUT->FUT)',
                            'corr_cit_past_div_out_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_in[file], future_out_to_to_div[file]),
                             'Correlation between div in(PAST->PAST) and div out(FUT->FUT)',
                             'corr_div_in_div_out_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_in[file], future_out_to_to_ref[file]),
                             'Correlation between cit(PAST->PAST) and refs(FUT->FUT)',
                             'corr_cit_past_ref_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_in[file], future_out_to_to_ref[file]),
                             'Correlation between div in(PAST->PAST) and refs(FUT->FUT)',
                             'corr_div_in_past_ref_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_out[file], future_from_to_cit[file]),
                             'Correlation between div out(PAST->PAST) and cit(FUT->PAST)',
                             'corr_div_out_past_cit_from_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_out[file], future_from_to_div[file]),
                             'Correlation between div out(PAST->PAST) and div in(FUT->PAST)',
                             'corr_div_out_past_div_in_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_out[file], future_from_to_cit[file]),
                             'Correlation between refs(PAST->PAST) and cit(FUT->PAST)',
                             'corr_ref_past_cit_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_out[file], future_from_to_div[file]),
                             'Correlation between refs(PAST->PAST) and div in(FUT->PAST)',
                             'corr_ref_past_div_in_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_in[file], future_from_to_cit[file]),
                             'Correlation between div in(PAST->PAST) and cit(FUT->PAST)',
                             'corr_div_in_past_cit_from_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_in[file], future_from_to_div[file]),
                             'Correlation between div in(PAST->PAST) and div in(FUT->PAST)',
                             'corr_div_in_past_div_in_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_in[file], future_from_to_cit[file]),
                             'Correlation between cit(PAST->PAST) and cit(FUT->PAST)',
                             'corr_cit_past_cit_fut_%s' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_in[file], future_from_to_div[file]),
                             'Correlation between cit(PAST->PAST) and div in(FUT->PAST)',
                             'corr_cit_past_div_in_fut_%s' % file)

    random_state = RandomState(seed=10)
    for file_to_save in files:
        print(file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_out[file], future_out_to_to_ref[file]),
                                      'Correlation between div out(PAST->PAST) and refs(FUT->FUT)',
                                      'corr_div_out_past_ref_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_out[file], future_out_to_to_div[file]),
                                      'Correlation between div out(PAST->PAST) and div out(FUT->FUT)',
                                      'corr_div_out_past_div_out_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_in[file], future_out_to_to_div[file]),
                                      'Correlation between cit(PAST->PAST) and div out(FUT->FUT)',
                                      'corr_cit_past_div_out_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_in[file], future_out_to_to_div[file]),
                                      'Correlation between div in(PAST->PAST) and div out(FUT->FUT)',
                                      'corr_div_in_div_out_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_in[file], future_out_to_to_ref[file]),
                                      'Correlation between cit(PAST->PAST) and refs(FUT->FUT)',
                                      'corr_cit_past_ref_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_in[file], future_out_to_to_ref[file]),
                                      'Correlation between div in(PAST->PAST) and refs(FUT->FUT)',
                                      'corr_div_in_past_ref_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_out[file], future_from_to_cit[file]),
                                      'Correlation between div out(PAST->PAST) and cit(FUT->PAST)',
                                      'corr_div_out_past_cit_from_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_out[file], future_from_to_div[file]),
                                      'Correlation between div out(PAST->PAST) and div in(FUT->PAST)',
                                      'corr_div_out_past_div_in_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_out[file], future_from_to_cit[file]),
                                      'Correlation between refs(PAST->PAST) and cit(FUT->PAST)',
                                      'corr_ref_past_cit_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_out[file], future_from_to_div[file]),
                                      'Correlation between refs(PAST->PAST) and div in(FUT->PAST)',
                                      'corr_ref_past_div_in_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_in[file], future_from_to_cit[file]),
                                      'Correlation between div in(PAST->PAST) and cit(FUT->PAST)',
                                      'corr_div_in_past_cit_from_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_div_in[file], future_from_to_div[file]),
                                      'Correlation between div in(PAST->PAST) and div in(FUT->PAST)',
                                      'corr_div_in_past_div_in_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_in[file], future_from_to_cit[file]),
                                      'Correlation between cit(PAST->PAST) and cit(FUT->PAST)',
                                      'corr_cit_past_cit_fut_%s' % file_to_save)
        corr_temporal_series_col_null(pool, random_state,
                                      *filter_const(temporal_series_in[file], future_from_to_div[file]),
                                      'Correlation between cit(PAST->PAST) and div in(FUT->PAST)',
                                      'corr_cit_past_div_in_fut_%s' % file_to_save)