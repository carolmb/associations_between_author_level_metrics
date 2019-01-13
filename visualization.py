from igraph import *
import numpy as np
import xnet
from collections import defaultdict
from scipy.sparse import dok_matrix
from scipy.spatial.distance import cosine
import sys
import matplotlib.pyplot as plt
import random

def cos_sim(v1,v2):
    s = 0.0
    
    if len(v1) and len(v2):    
        for k,v in v1.items():
            if k in v2:
                s += v*v2[k]

        if s:
            v1 = list(v1.values())
            s_v1 = np.sum(np.asarray(v1)**2)
            v2 = list(v2.values())
            s_v2 = np.sum(np.asarray(v2)**2)
            cos = s/(np.sqrt(s_v1)*np.sqrt(s_v2))

            return cos

    return 0.0

def filter_window(net,begin,delta):
    valid_vtxs = net.vs.select(year_ge=begin,year_lt=begin+delta)
    subgraph = net.subgraph(valid_vtxs)
    return subgraph

def author_colabs_author(net,begin,delta,valid_authors):
    print('colabs begin',begin,'delta',delta)
    papers = net.vs
    colabs = defaultdict(lambda:0)
    for paper in papers:
        if paper['year'] >= begin and paper['year'] < begin+delta:
            # print(paper['year'])
            authors = paper['authors_idx'].split(',')
            authors = [a for a in authors if a in valid_authors]
            N = len(authors)
            for a0 in authors:
                for a1 in authors:
                    if a0 == a1:
                        break
                    colabs[frozenset({a0,a1})] += 1/N
    print('colabs',len(colabs))
    colabs = dict(colabs)
    print(colabs)
    return colabs

def author_cites_author(net,begin,delta):
    print('authors_cites_author')

    small_net = filter_window(net,begin,delta)
    authors_idx = small_net.vs['authors_idx']
    authors = []
    for a_list in authors_idx:
        a_list = a_list.split(',')
        authors += a_list

    try:
        authors.remove('')
    except:
        print('all valid authors')
        pass

    authors = np.asarray(authors)
    unique_authors,count_authors = np.unique(authors,return_counts=True)

    '''
    only the authors with 20 or more publications
    '''
    valid_idxs = count_authors[count_authors>10]
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
    unique_edges = edges[:,0]
    count = edges[:,1]
    total = len(unique_edges)
    print('total of citations',total)
    
    citation_net.add_edges(unique_edges)
    citation_net.es['weight'] = count

    citation_net = citation_net.components().giant()
    
    return citation_net, unique_authors

'''
generate the vector presentation
'''
def repre_attribute(net,mode='out'): # cited or be cited
    N = len(net.vs)
    for v in net.vs:
        repre = dict()
        edges_idxs = net.incident(v,mode=mode) # TODO verify if it is really 'out'
        for e_idx in edges_idxs:
            edge = net.es[e_idx]
            a0,a1 = edge.tuple
            # print(a0,a1)
            weight = edge['weight']

            if not v.index == a0:
                pos = a0
            else:
                pos = a1
            repre[pos] = weight

        v['repre'] = repre

    print('repre attributes',net.vs.attributes())

    # for idx,row in enumerate(repre):
    #     net.vs[idx]['repre'] = row
    #     if idx%1000 == 0:
    #         print(idx,end='\r')
    return net

def calculate_sim(net,valid_pairs):
    valid_pairs = set(valid_pairs)
    sims = dict()
    authors = net.vs
    error = 0
    i = 0
    # print('valid_pairs',valid_pairs)
    for a0,a1 in valid_pairs:
        try:
            v0 = authors.find(name_eq=a0)['repre']
            v1 = authors.find(name_eq=a1)['repre']
            # print(a0,a1)
            s = cos_sim(v0,v1)
            # print('sim',s)
            sims[(a0,a1)] = s
            del v0
            del v1
            i += 1
            if i%100 == 0:
               print(i,end='\r')
        except Exception as e:
            error += 1
            # print(e,end='\r')
    print('total missing',error)
    return sims

def plot_sim_vs_colab(sims,forces,begin):
    point_x = []
    point_y = []

    keys1 = set(sims.keys())
    keys1 = set([frozenset(k) for k in keys1])
    keys2 = set(forces.keys())
    keys = keys1 & keys2
    print('total force',len(keys2),'commum keys',len(keys))

    # keys = random.sample(keys,100000)

    for key in keys:
        point_y.append(forces[key])

        key = tuple(key)
        try:
            point_x.append(sims[tuple(key)])
        except:
            point_x.append(sims[(key[1],key[0])])
    
    point_x = np.asarray(point_x)
    point_y = np.asarray(point_y)

    idxs = np.argsort(point_y)[:-100]
    print(idxs)
    point_x = point_x[idxs]
    point_y = point_y[idxs]

    plt.scatter(point_x,point_y,alpha=0.5)
    # heatmap, xedges, yedges = np.histogram2d(point_x,point_y,bins=50)
    # extent = [xedges[0],xedges[-1],yedges[0],yedges[-1]]

    # plt.imshow(heatmap.T,extent=extent,origin='lower')
    plt.savefig('imgs/sim_cited_vs_colab'+str(begin)+'.png')
    plt.clf()

