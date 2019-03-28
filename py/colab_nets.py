#!/usr/bin/env python
# coding: utf-8

# In[2]:


from igraph import *
import xnet
import util
# import math
import numpy as np
import time
import pickle

# In[3]:


net = xnet.xnet2igraph('../data/citation_net_ge_1990.xnet')

vcount = net.vcount()
net.vs['numeric_id'] = range(vcount)

# In[5]:

min_year = 1990
max_year = 2010
delta = 5

authors_idx = net.vs['authors_idxs']
authors = []
for a_list in authors_idx:
    a_list = a_list.split(',')
    authors += a_list

authors = set([a for a in authors if not a == ''])
print('total number of authors',len(authors))

# In[ ]:


for year in range(2010,max_year+1):
    print('current year',year)
    t0 = time.time()
    colab_dict, paper_dict, all_authors  = util.author_colabs_author(net,year,delta,authors)
    t1 = time.time()
    print('time util.author_colabs_author in seconds',t1-t0)

    #idxs = set()
    #edges = []
    #weights_basic = []
    #weights_comb = []
    #papers = []

    #keys = set(colab_dict.keys())
    t0 = time.time()
    idxs = list(all_authors)
    del all_authors
    edges = list(colab_dict.keys())
    weights_basic = np.asarray(list(colab_dict.values()))
    del colab_dict
    t1 = time.time()
    print('time idxs,edges,weights_basic in seconds',t1-t0)

    papers = list(paper_dict.values())
    del paper_dict
    papers_lens = np.asarray([len(ps) for ps in papers])
    weights_comb = papers_lens*weights_basic
    papers = [str(ps) for ps in papers]
    t2 = time.time()
    print('time papers,papers_len,weights_comb time in seconds',t2-t1)

    '''
    for key in keys:
        idxs.add(key[0])
        idxs.add(key[1])
        edges.append(key)
        weights_basic.append(colab_dict[key])
        total_colabs = len(paper_dict[key])
        weights_comb.append(colab_dict[key]*total_colabs)
        papers.append(str(paper_dict[key]))
        del colab_dict[key]
        del paper_dict[key]'''

    #idxs = list(idxs)
    print('total of authors',len(idxs),'total of colabs',len(edges))

    colab_net = Graph()
    colab_net.add_vertices(len(idxs))
    colab_net.vs['name'] = idxs
    colab_net.add_edges(edges)
    colab_net.es['weight_basic'] = weights_basic
    colab_net.es['weight_comb'] = weights_comb
    colab_net.es['papers'] = papers
    t3 = time.time()
    print('time creating graph',t3-t2)
    xnet.igraph2xnet(colab_net,'colabs/original/colab_'+str(year)+'_'+str(year+delta)+'_test')
    t4 = time.time()
    print('time saving',t4-t3)
    print()

    break

# In[ ]:





# In[ ]:




