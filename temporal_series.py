#!/usr/bin/env python
# coding: utf-8

import xnet
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from util import save, load
from datetime import datetime
from scipy.stats import spearmanr
from collections import defaultdict
from numpy.random import RandomState

# random_state = RandomState(seed=10)
# print(random_state.get_state()[1][0])

import warnings
warnings.filterwarnings("error")


def get_temporal_series(valid_authors, data):
    Y = []
    for author in valid_authors:
        history = data[author]
        y = []
        for year in range(1995, 2011):
            try:
                value = history[str(year)]
            except:
                value = 0
            y.append(value)
        Y.append(np.asarray(y))
    Y = np.asarray(Y)
    return Y


'''
autores válidos seguindo critério frouxo
'''


def read_valid_authors():
    name_to_authors = dict()
    file = 'data2/valid_authors_min_criteria_in_out_10_10_temp_v3.txt'
    valid_authors = open(file, 'r').read().split("\n")[:-1]
    name_to_authors[file[:-4]] = valid_authors
    return name_to_authors


def get_authors_by_percentile(author_values, key_header):
    authors = []
    values = []
    for k, v in author_values.items():
        authors.append(k)
        values.append(v)

    authors = np.asarray(authors)
    values_space = np.percentile(values, [0, 25, 50, 75, 100])
    values_space[-1] += 1
    print('percentiles', values_space)
    values = np.asarray(values)
    author2class = np.searchsorted(values_space, values, 'right')

    authors_by_class = dict()
    unique_values = np.unique(author2class)
    for c in unique_values:
        authors_by_class[key_header + str(c)] = authors[author2class == c]
        temp = values[author2class == c]
        print(c, min(temp), max(temp))
    for key_class, authors in authors_by_class.items():
        print(key_class, len(authors))
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
    true_corr = spearmanr(X, Y)[0]

    n = len(X)
    idxs1 = np.arange(n)
    idxs2 = np.arange(n)
    corrs = []

    dt = datetime.now()
    # print(dt)
    random_state = RandomState(seed=dt.microsecond)
    for i in range(iters):
        random_state.shuffle(idxs1)
        random_state.shuffle(idxs2)
        corr = spearmanr(X[idxs1], Y[idxs2])[0]
        corrs.append(corr)
    corrs = np.asarray(corrs)
    nonzero = np.count_nonzero(np.abs(corrs) > np.abs(true_corr))
    p_val = nonzero / iters

    hist, _ = np.histogram(corrs, bins=bins)

    return np.asarray([true_corr, p_val, hist], dtype=object)


def get_author_paper_example(data, author_id):
    found = False
    for paper in data.vs:
        i = 0
        for author in paper['authors_idxs'].split(','):
            if author == author_id:

                found = True
                print(author_id+' '+paper['title']+' '+paper['authors_names'].split(';')[i])
                break
            i += 1
        if found:
            break


def corr_temporal_series_curves_samples(data, pool, X, Y, Z, authors, x_label, y_label, filename):

    output = pool.map(null_model, list(zip(X, Y)))
    output = np.asarray(output)

    corrs = output[:, 0]
    p_vals = output[:, 1]

    # idxs_le = p_vals > 0.05
    # corr_le = corrs[idxs_le]
    # authors_le = authors[idxs_le]
    # argsort_le = np.argsort(-np.abs(corr_le))
    # label_le = 'p-value > 0.05'

    idxs_g_pos = np.logical_and(p_vals <= 0.05, corrs >= 0)
    corr_g_pos = corrs[idxs_g_pos]
    authors_pos = authors[idxs_g_pos]
    Z_pos = Z[idxs_g_pos]
    argsort_pos = np.argsort(-np.abs(corr_g_pos))
    label_g_pos = 'p-value <= 0.05 (corr >= 0)'

    idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
    corr_g_neg = corrs[idxs_g_neg]
    authors_neg = authors[idxs_g_neg]
    Z_neg = Z[idxs_g_neg]
    argsort_neg = np.argsort(-np.abs(corr_g_neg))
    label_g_neg = 'p-value <= 0.05 (corr < 0)'

    # # CURVES
    # corr_le_curves_x = X[idxs_le]
    # corr_le_curves_x = corr_le_curves_x[argsort_le]
    # corr_le_curves_y = Y[idxs_le]
    # corr_le_curves_y = corr_le_curves_y[argsort_le]
    # authors_le = authors_le[argsort_le]

    # for

    corr_g_pos_curves_x = X[idxs_g_pos]
    corr_g_pos_curves_x = corr_g_pos_curves_x[argsort_pos]
    corr_g_pos_curves_y = Y[idxs_g_pos]
    corr_g_pos_curves_y = corr_g_pos_curves_y[argsort_pos]
    authos_pos = authors_pos[argsort_pos]
    Z_pos = Z_pos[argsort_pos]

    corr_g_neg_curves_x = X[idxs_g_neg]
    corr_g_neg_curves_x = corr_g_neg_curves_x[argsort_neg]
    corr_g_neg_curves_y = Y[idxs_g_neg]
    corr_g_neg_curves_y = corr_g_neg_curves_y[argsort_neg]
    authors_neg = authors_neg[argsort_neg]
    Z_neg = Z_neg[argsort_neg]

    time = np.arange(1995, 2008)
    print('pos')
    for x, y, n, a, c in zip(corr_g_pos_curves_x[:3], corr_g_pos_curves_y[:3], Z_pos[:3], authos_pos[:3], corr_g_pos[argsort_pos]):
        plt.plot(time, x, label=x_label)
        plt.plot(time, y, label=y_label)
        plt.plot(time, n, label='cits without self-cits')
        plt.title('\npos corr %.2f, new corr %.2f author %a' % (c, spearmanr(x, n)[0], a))
        plt.legend()
        plt.savefig('corr_analysis/%s_pos_corr_author_%s_2.pdf' % (filename, a))
        plt.clf()
        get_author_paper_example(data, a)

    print('neg')
    for x, y, n, a, c in zip(corr_g_neg_curves_x[:3], corr_g_neg_curves_y[:3], Z_neg[:3], authors_neg[:3], corr_g_neg[argsort_neg]):
        plt.plot(time, x, label=x_label)
        plt.plot(time, y, label=y_label)
        plt.plot(time, n, label='cits without self-cits')
        plt.title('\nneg corr %.2f, new corr %.2f, author %a' % (c, spearmanr(x, n)[0], a))
        plt.legend()
        plt.savefig('corr_analysis/%s_neg_corr_author_%s_2.pdf' % (filename, a))
        plt.clf()
        get_author_paper_example(data, a)


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
    label_g_pos = r'p-value $\leq$ 0.05 (corr $\geq$ 0)' # <= >=

    idxs_g_neg = np.logical_and(p_vals <= 0.05, corrs < 0)
    corr_g_neg = corrs[idxs_g_neg]
    label_g_neg = r'p-value $\leq$ 0.05 (corr < 0)'

    # HIST STACKED
    # plt.title(title)
    plt.xlim(-1, 1)
    plt.hist([corr_g_pos, corr_g_neg, corr_le],
             bins=bins, alpha=0.6, stacked=True,
             color=['orange', 'blue', 'gray'],
             label=[label_g_pos, label_g_neg, label_le])
    plt.legend(loc="upper right", prop={'size': 14})
    plt.tick_params(labelsize=13)

    # CORR INFOS MEAN AND STD
    mu = np.nanmean(corrs)
    sigma = np.nanstd(corrs)
    q_index = len(corr_g_pos) / len(corr_g_neg) if len(corr_g_neg) > 0 else np.nan
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu,),
        r'$\sigma=%.2f$' % (sigma,),
        r'q=%.2f' % (q_index,)))
    props = dict(boxstyle='round', facecolor='gray', alpha=0.3)
    ax = plt.gca()
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.text(0.45, 0.1, "%.2f%%" % (100 * len(corr_le) / len(corrs)),
            color='gray', transform=ax.transAxes, fontsize=14)
    ax.text(0.85, 0.1, "%.2f%%" % (100 * len(corr_g_pos) / len(corrs)),
            color='orange', transform=ax.transAxes, fontsize=14)
    ax.text(0.05, 0.1, "%.2f%%" % (100 * len(corr_g_neg) / len(corrs)),
            color='blue', transform=ax.transAxes, fontsize=14)

    # NULL MODEL PLOT1

    x_bin_center = [(bins[i] + bins[i + 1]) / 2 for i in range(n_bins - 1)]
    plt.errorbar(x_bin_center, hist_ave, yerr=hist_std, color='red')
    plt.tight_layout()
    # plt.savefig('temporal_series_data/v5/' + file + '.pdf')
    plt.savefig('sinatra/' + file + '.pdf')
    plt.clf()
    return corrs


def filter_const(X, Y):
    X2 = []
    Y2 = []
    for x, y in zip(X, Y):
        if np.count_nonzero(y[:-3] == y[0]) == (len(y)-3) or np.count_nonzero(x[:-3] == x[0]) == (len(x)-3):
            continue # tira todo mundo que é um vetor constante

        X2.append(x[:-3])
        Y2.append(y[:-3])
    X2 = np.asarray(X2)
    Y2 = np.asarray(Y2)
    return X2, Y2


def authors_groups(files_valid_authors):
    paper_count = defaultdict(lambda: 0)

    i = 0
    for key, valid_authors in files_valid_authors.items():
        valid_authors.remove('')
        for paper in data.vs:
            if paper['year'] < 1991 or paper['year'] > 2010:
                continue
            i += 1
            authors_idxs = paper['authors_idxs'].split(',')
            for author in authors_idxs:
                if author in valid_authors:
                    paper_count[author] += 1
            if i % 100000 == 0:
                print(i)
        save(paper_count, 'authors_paper_count_1991.json')
        break


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
    authors_out_per_paper = load('data2/authors_out_freq_by_paper_v3.json')
    authors_in_per_paper = load('data2/authors_in_freq_by_paper_v3.json')  # citações
    authors_out = load('data2/authors_out_freq_v3.json')
    authors_in = load('data2/authors_in_freq_v3.json')  # citações
    authors_number_of_papers = load('data2/number_of_papers_v3.json')

    # ESSES SÃO OS REGISTROS DO LADO DIREITO
    cit_from_abs_citations = load('data2/cit_from_citations_v3.json')
    cit_from_citations = load('data2/cit_from_citations_per_paper_v3.json') # TODO aqui é _per_paper
    cit_from_diversity = load('data2/cit_from_diversity_v3.json')
    out_to_to_citations = load('data2/out_to_to_citations_per_paper_v3.json')
    out_to_to_diversity = load('data2/out_to_to_diversity_v3.json')
    authors_number_of_papers_fut = load('data2/number_of_papers_fut_fut_v3.json')

    authors_in_div = load('data2/authors_in_div_v3.json')
    authors_out_div = load('data2/authors_out_div_v3.json')

    files_valid_authors = read_valid_authors()
    print(files_valid_authors.keys())

    data = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')
    authors_groups(files_valid_authors)

    paper_count_1991 = load('authors_paper_count_1991.json')

    authors_by_paper_count_1991 = get_authors_by_percentile(paper_count_1991, 'paper_')

    files_valid_authors = {**authors_by_paper_count_1991}

    files = list(files_valid_authors.keys())
    print(files)
    temporal_series_div_out = dict()
    temporal_series_div_in = dict()
    temporal_series_in = dict()
    temporal_series_in_abs = dict()
    temporal_series_out = dict()
    temporal_series_numb_papers = dict()

    future_from_to_cit_abs = dict()
    future_from_to_cit = dict()
    future_from_to_div = dict()
    future_out_to_to_ref = dict()
    future_out_to_to_div = dict()
    future_number_of_papers = dict()

    for file, valid_authors in files_valid_authors.items():

        temporal_series_div_out[file] = get_temporal_series(valid_authors, authors_out_div)
        temporal_series_div_in[file] = get_temporal_series(valid_authors, authors_in_div)
        temporal_series_in[file] = get_temporal_series(valid_authors, authors_in_per_paper)
        temporal_series_in_abs[file] = get_temporal_series(valid_authors, authors_in)
        temporal_series_out[file] = get_temporal_series(valid_authors, authors_out_per_paper)
        temporal_series_numb_papers[file] = get_temporal_series(valid_authors, authors_number_of_papers)

        future_from_to_cit_abs[file] = get_temporal_series(valid_authors, cit_from_abs_citations)
        future_from_to_cit[file] = get_temporal_series(valid_authors, cit_from_citations)
        future_from_to_div[file] = get_temporal_series(valid_authors, cit_from_diversity)
        future_out_to_to_ref[file] = get_temporal_series(valid_authors, out_to_to_citations)
        future_out_to_to_div[file] = get_temporal_series(valid_authors, out_to_to_diversity)
        future_number_of_papers[file] = get_temporal_series(valid_authors, authors_number_of_papers_fut)

    # plot_history(files_valid_authors, data, 'paper', 'authors_paper_count_history.pdf')
    # plot_history(files_valid_authors, data, 'cit', 'authors_cit_history.pdf')

    pool = multiprocessing.Pool(processes=8)

    # random_state = RandomState(seed=9)
    for file in files:
        print(file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_out[file], future_from_to_cit[file]),
                             'Correlation between div out(PAST->PAST) and cit(FUT->PAST)',
                             'corr_div_out_past_cit_from_fut_%s_1991' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_out[file], future_from_to_div[file]),
                             'Correlation between div out(PAST->PAST) and div in(FUT->PAST)',
                             'corr_div_out_past_div_in_fut_%s_1991' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_div_in[file], future_from_to_cit[file]),
                             'Correlation between div in(PAST->PAST) and cit(FUT->PAST)',
                             'corr_div_in_past_cit_from_fut_%s_1991' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_in[file], future_from_to_div[file]),
                             'Correlation between cit(PAST->PAST) and div in(FUT->PAST)',
                             'corr_cit_past_div_in_fut_%s_1991' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_numb_papers[file], future_from_to_cit[file]),
                             'Correlation between papers(PAST->PAST) and cit(FUT->PAST)',
                             'corr_papers_past_cit_from_fut_%s_1991' % file)

        corr_temporal_series(pool, *filter_const(temporal_series_numb_papers[file], future_from_to_div[file]),
                             'Correlation between papers(PAST->PAST) and div in(FUT->PAST)',
                             'corr_papers_past_div_in_fut_%s_1991' % file)
