from igraph import *
import numpy as np
import xnet
from collections import defaultdict
import sys

def filter_window(net,begin,delta):
    valid_vtxs = net.vs.select(year_ge=begin,year_lt=begin+delta)
    subgraph = net.subgraph(valid_vtxs)
    return subgraph

def author_cites_author(net,begin,delta):
    print('authors_cites_author')

    small_net = filter_window(net,begin,delta)
    authors_idx = small_net.vs['authors_idx']
    authors = []
    for a_list in authors_idx:
        a_list = a_list.split(',')
        authors += a_list

    authors = [a for a in authors if not a == '']

    authors = np.asarray(authors)
    unique_authors,count_authors = np.unique(authors,return_counts=True)

    # print(len(unique_authors))
    # print(np.sort(count_authors)[-100:])

    '''
    only the authors with 20 or more publications
    '''
    valid_idxs = count_authors>20
    unique_authors = unique_authors[valid_idxs]

    print('total of authors',len(unique_authors))
    citation_net = Graph(directed=True)
    citation_net.add_vertices(len(unique_authors))
    citation_net.vs['name'] = unique_authors

    papers = small_net.vs
    print('total of papers',small_net.vcount())
    edges = defaultdict(lambda:0)
    for p in papers:
        author_idxs = p['authors_idx'].split(',')
        author_idxs = [a for a in author_idxs if a in unique_authors]
        if len(author_idxs) > 0:
            # print('author_idxs',author_idxs)

            citing = small_net.neighbors(p,mode='out')
            citing = [small_net.vs[c] for c in citing]

            cited_idx = []
            for paper_cited in citing:
                idxs = paper_cited['authors_idx'].split(',')
                idxs = [i for i in idxs if i in unique_authors]
                cited_idx += idxs

            if len(cited_idx) > 0:
                # print('cited_idx',cited_idx)

                for a in author_idxs:
                    for c in cited_idx:
                        if a == c:
                            continue
                        edges[(a,c)] += 1

    edges = np.asarray(list(edges.items()))
    print(edges.shape)
    unique_edges = edges[:,0]
    count = edges[:,1]
    total = len(unique_edges)
    print('total of citations',total)

    citation_net.add_edges(unique_edges)
    citation_net.es['weight'] = count

    return citation_net, unique_authors

