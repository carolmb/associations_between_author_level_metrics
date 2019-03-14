from igraph import *
import xnet
import numpy as np

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
    print('colabs begin',begin,'delta',delta)
    papers = net.vs
    colabs = defaultdict(lambda:0)
    papers_from_colab = defaultdict(lambda:[])
    for paper in papers:
        title = paper['title']
        abstract = paper['abstract']

        if paper['year'] >= begin and paper['year'] < begin+delta:
            authors = paper['authors_idxs'].split(',')
            authors = [a for a in authors if a in valid_authors]
            N = len(authors)
            for a0 in authors:
                for a1 in authors:
                    if a0 == a1:
                        break
                    colabs[frozenset({a0,a1})] += 1/(N-1)
                    papers_from_colab[frozenset({a0,a1})].append(paper['id'])

    print('colabs',len(colabs))
    colabs = dict(colabs)
    papers_from_colab = dict(papers_from_colab)
    # print(colabs)
    return colabs,papers_from_colab


