#!/usr/bin/env python
# coding: utf-8

# In[2]:

from igraph import *
import xnet
import util
import numpy as np
import time

# In[3]:


net = xnet.xnet2igraph('../data/citation_network_ge1990_pacs.xnet')

if not 'numeric_id' in net.vs.attributes():
    net = util.get_net(net)
    print("Não tem numeric_id")
    vcount = net.vcount()
    net.vs['numeric_id'] = list(range(vcount))
    xnet.igraph2xnet(net,'../data/citation_network_ge1990_pacs.xnet')


# net = Graph()
# net.add_vertices(5)
# net.vs['numeric_id'] = list(range(5))
# net.vs['authors_names'] = ['A;B;C','B;C;D','C;E','A;E']
# net.vs['authors_idxs'] = ['1,2,3','2,3,4','3,5','1,5']
# net.vs['year'] = [2009,2009,2008,2004,2004]
# net.vs['times_cited'] = [1,30,25,20,5]


# In[5]:
'''
min_year = 1990
max_year = 2010
delta = 3

authors_idx = net.vs['authors_idxs']
authors = []
for a_list in authors_idx:
    a_list = a_list.split(',')
    authors += a_list

authors = set([int(a) for a in authors if not a == ''])
print('total number of authors',len(authors))

temp = 0
for a in authors:
    print(a,end=' ')
    temp += 1
    if temp > 10:
        break
print()

# In[ ]:

for year in range(min_year,max_year+1):
    print('current year',year)
    t0 = time.time()
    colab_dict, paper_dict, all_authors  = util.author_colabs_author(net,year,delta,authors)
    t1 = time.time()
    print('time util.author_colabs_author in seconds',t1-t0)

    t0 = time.time()
    idxs = list(all_authors)
    idxs = [str(i) for i in idxs]
    del all_authors
    edges = list(colab_dict.keys())
    edges = [(str(a),str(b)) for a,b in edges]
    weights_basic = np.asarray(list(colab_dict.values()))
    del colab_dict
    t1 = time.time()
    print('time idxs,edges,weights_basic in seconds',t1-t0)

    papers_by_colab = list(paper_dict.values())
    del paper_dict
    papers_lens = np.asarray([len(ps) for ps in papers_by_colab])
    weights_comb = papers_lens*weights_basic
    weights_comb_log = np.log(papers_lens)*weights_basic + 1
    papers_by_colab = [str(ps) for ps in papers_by_colab]
    t2 = time.time()
    print('time papers_by_colab,papers_len,weights_comb time in seconds',t2-t1)

    print('total of authors',len(idxs),'total of colabs',len(edges))

    colab_net = Graph()
    colab_net.add_vertices(len(idxs))
    colab_net.vs['name'] = idxs
    colab_net.add_edges(edges)
    colab_net.es['weight_basic'] = weights_basic
    colab_net.es['weight'] = weights_comb
    colab_net.es['weight_comb_log'] = weights_comb_log
    colab_net.es['papers'] = papers_by_colab
    t3 = time.time()
    print('time creating graph',t3-t2)
    xnet.igraph2xnet(colab_net,'colabs/original/colab_'+str(year)+'_'+str(year+delta)+'.xnet')
    t4 = time.time()
    print('time saving',t4-t3)
    print()

'''
# In[ ]:





# In[ ]:




