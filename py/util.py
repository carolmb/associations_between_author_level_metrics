from igraph import *
import xnet
import numpy as np
import time
import itertools

'''
Clear database
'''
def get_net():
    net = xnet.xnet2igraph('../data/citation_network.xnet')

    print(net.vs.attributes())

    # titles = net.vs['title']
    # unique_titles,count_titles = np.unique(titles,return_counts=True)

    print(net.vcount(),net.ecount())

    invalid_vtxs = net.vs.select(authors_idxs_eq='')
    net.delete_vertices(invalid_vtxs)
    print(net.vcount(),net.ecount())

    invalid_vtxs = net.vs.select(abstract_eq='')
    net.delete_vertices(invalid_vtxs)
    print(net.vcount(),net.ecount())

    invalid_vtxs = net.vs.select(title_eq='')
    net.delete_vertices(invalid_vtxs)
    print(net.vcount(),net.ecount())

    invalid_vtxs = net.vs.select(title_eq='CORRECTION')
    net.delete_vertices(invalid_vtxs)
    print(net.vcount(),net.ecount())

    invalid_vtxs = net.vs.select(year_lt=1989)
    net.delete_vertices(invalid_vtxs)
    print(net.vcount(),net.ecount())

    '''titles = net.vs['title']
    unique_titles,count_titles = np.unique(titles,return_counts=True)
    for u,c in zip(unique_titles,count_titles):
        if c == 2:
            vs = net.vs.select(title_eq=u)
            for v in vs:
                print(v['title'],v['abstract'],v['authors_idx'])
            print()
    '''

    return net


def author_colabs_author(net,begin,delta,valid_authors):
    #begin_time = time.time()
    #print('colabs begin',begin_time,'delta',delta)

    colabs = defaultdict(lambda:0)
    papers_from_colab = defaultdict(lambda:[])
    all_authors = set()

    #print(net.vs.attributes())
    papers = net.vs.select(year_ge=begin,year_le=begin+delta)
    #papers = papers.select(year_lt=begin+delta)
    #print('total of papers:::::',len(papers))

    #middle_time = time.time()

    #print('middle delta',middle_time-begin_time)

    #print('total of papers',len(papers))
    i = 0
    for paper in papers:

        i += 1
        if i%10000 == 0:
            print(i)

        
        authors = paper['authors_idxs'].split(',')
        authors = [int(a) for a in authors if int(a) in valid_authors]
        all_authors |= set(authors)
        authors = sorted(authors)
        N = len(authors)
        paper_id = paper['numeric_id']

        combinations = itertools.combinations(authors,2)

        for key in combinations:
            colabs[key] += 1/(N-1)
            papers_from_colab[key].append(paper_id)

    #end_time = time.time()
    #print('end delta',end_time-middle_time)

    print('colabs',len(colabs))
    colabs = dict(colabs)
    papers_from_colab = dict(papers_from_colab)
    return colabs,papers_from_colab,all_authors


