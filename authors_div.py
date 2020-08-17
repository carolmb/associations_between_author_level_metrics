import xnet
import json
import glob
import util

import numpy as np
import matplotlib.pyplot as plt

from igraph import *
from util import save,load
from scipy.stats import pearsonr
from collections import defaultdict
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms,gridspec

attr_pacs = util.get_attr_pacs() # nome dos atributos dos vértices que contem os códigos PACS
pac_list = util.get_pac_list() # lista de códigos válidos


def get_citation_future(paper,future_interval, MODE):
    citations = []
    for neighbor in paper.neighbors(mode=MODE):
        if future_interval[0] <= neighbor['year'] <= future_interval[1]:
            citations.append(neighbor)
    return citations
    

def get_freq_of_future(data,MODE,delta_past,delta_future):

    history = defaultdict(lambda: defaultdict(lambda: []))
    papers = defaultdict(lambda: defaultdict(lambda: 0))
    year_begin = 1986
    year_end = 2007

    for i, year in enumerate(range(year_begin,year_end+1)):
        subset = data.vs.select(year_ge=year, year_le=year+delta_past)
        print('to interval', year, year+delta_past)
        future_interval = (year+delta_past+1, year+delta_past+1+delta_future)
        print('from interval', future_interval)
            
        for paper in subset:
            authors_idxs = paper['authors_idxs'].split(',')
            citations = get_citation_future(paper, future_interval, MODE)
            for author in authors_idxs:
                history[author][year+delta_past] += citations
                papers[author][year+delta_past] += 1
        print()
    return history, papers


def get_freq_of_citall(data, MODE, delta_past, delta_future):

    history = defaultdict(lambda: defaultdict(lambda: []))
    papers = defaultdict(lambda: defaultdict(lambda: 0))
    year_begin = 1986
    year_end = 2007
            
    for i,year in enumerate(range(year_begin,year_end+1)):
        subset = data.vs.select(year_ge=year,year_le=year+delta_past)
        print('to interval',year,year+delta_past)
        future_interval = (year,year+delta_past+1+delta_future)
        print('from interval',future_interval)
    
        for paper in subset:
            authors_idxs = paper['authors_idxs'].split(',')
            citations = get_citation_future(paper,future_interval,MODE)
            for author in authors_idxs:
                history[author][year+delta_past] += citations
                papers[author][year+delta_past] += 1
        print()
    return history, papers


def get_freq_of_futute_future(data, MODE, delta_past, delta_future):

    history = defaultdict(lambda: defaultdict(lambda:[]))
    papers = defaultdict(lambda: defaultdict(lambda: 0))
    year_begin = 1986
    year_end = 2007

    for i,year in enumerate(range(year_begin, year_end+1)):
        subset = data.vs.select(year_ge=year+delta_past+1, year_le=year+delta_past+1+delta_future)
        print('to interval', year, year+delta_past)
        future_interval = (year+delta_past+1, year+delta_past+1+delta_future)
        print('from interval', future_interval)
        
        for paper in subset:
            authors_idxs = paper['authors_idxs'].split(',')
            
            citations = get_citation_future(paper,future_interval,MODE)
            for author in authors_idxs:
                history[author][year+delta_past] += citations
                papers[author][year+delta_past] += 1
        print()
    return history, papers


def get_div_paper(papers,pac_nets,get_pacs):
    comms_hist = []
    for paper in papers:
        comms = util.get_pacs_paper_published(paper,None)
        comms_hist += comms
    dist = defaultdict(lambda:0)
    for comms,n_comms in comms_hist:
        for comm in comms:
            dist[comm] += 1/n_comms
    if len(dist) == 0:
        return 0
    dist = list(dist.values())
    dist = np.asarray(dist)
    dist = dist/np.sum(dist)
    div = util.get_div(dist)
    return div
        

def get_citations(history, papers):
    author_citations = defaultdict(lambda: defaultdict(lambda: 0))
    for author, hist in history.items():
        for year, citations in hist.items():
            author_citations[author][year] = len(citations)/papers[author][year]

    return author_citations
            

def get_div(history, pac_nets):
    author_divs = defaultdict(lambda: defaultdict(lambda: 0))
    for author, hist in history.items():
        for year, citations in hist.items():
            author_divs[author][year] = get_div_paper(citations, pac_nets, util.get_pacs_in)
    return author_divs


if __name__ == '__main__':
    data = xnet.xnet2igraph('data/citation_network_ge1985_pacs.xnet')

    filenames = sorted(glob.glob('data/pacs/2lvls/*_delta4_multilevel2.xnet'))
    pac_nets = []
    for filename in filenames:
        net = xnet.xnet2igraph(filename)
        pac_nets.append(net)

    history_out_filename = 'data2\\authors_pac_out_2lvls_2multi.json'
    history_in_filename = 'data2\\authors_pac_in_2lvls_2multi.json'
    history_filename = 'data2\\authors_pac_2lvls_2multi.json'

    # util.get_pac_comm_freq(data,pac_nets,util.get_pacs_out,4,history_out_filename)
    # util.get_pac_comm_freq(data,pac_nets,util.get_pacs_in,4,history_in_filename)
    # placeholder = get_num_citations(data,'authors_citations.json')
    # util.get_pac_comm_freq(data,pac_nets,util.get_pacs_paper_published,4,history_filename)

    history_in = load(history_in_filename)
    history_out = load(history_out_filename)
    
    # authors_out_div = util.authors_metric(history_out,util.get_div) # diversidade das publicações citadas (author out)
    # save(authors_out_div,'data2\\authors_out_div.json')
    # authors_in_div = util.authors_metric(history_in,util.get_div) # diversidade das publicações que citam o autor (author in)
    # save(authors_in_div,'data2\\authors_in_div.json')

    # authors_in_sim = util.authors_metric(history_in,util.get_symmetry)
    # save(authors_in_sim,'data2\\authors_in_sym.json')
    # authors_out_sim = util.authors_metric(history_out,util.get_symmetry)
    # save(authors_out_sim,'data2\\authors_out_sym.json')

    # from = 5, to = 3
    print('in')
    #cit_from, papers = get_freq_of_future(data, IN, 4, 2) #'data/future_citations3.json')
    #cit_from_div = get_div(cit_from, pac_nets)
    #cit_from_cit = get_citations(cit_from, papers)
    #save(cit_from_div, 'data2\\cit_from_diversity.json')
    #save(cit_from_cit, 'data2\\cit_from_citations_per_paper.json')

    # from = 5, to = 3
    # all 5 + 3
    print('in citall')
    cit_from, papers = get_freq_of_citall(data, IN, 4, 2) #'data/future_citations3.json')
    #cit_from_div = get_div(cit_from, pac_nets)
    cit_from_cit = get_citations(cit_from, papers)
    #save(cit_from_div, 'data2\\cit_all_diversity.json')
    save(cit_from_cit, 'data2\\cit_all_citations_per_paper.json')

    # from = 3, to = 3
    print('out')
    cit_from, papers = get_freq_of_futute_future(data, OUT, 2, 2) #'data/future_citations3.json')
    #cit_from_div = get_div(cit_from, pac_nets)
    cit_from_cit = get_citations(cit_from, papers)
    #save(cit_from_div, 'data2\\out_to_to_diversity.json')
    save(cit_from_cit, 'data2\\out_to_to_citations_per_paper.json')


    '''
    from = 5
    to = 3

    [_from_] [_to_]

    from:
    citacoes de papers de [_from_]  em [_from_] ok
    div in/out de papers de [_from_] em [_from_]

    to:
    citFrom: citacoes de papers de [_from_]   em [_to_] ok
    div inFrom: div in de papers de [_from_]  em [_to_] ok

    citAll: citacoes de papers de [_from_]   em  [_from_][_to_]
    div inAll: div in de papers de [_from_]  em [_from_][_to_]

    div out: div out de papers de [__to__]  em [_to_]
    '''