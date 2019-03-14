#!/usr/bin/env python
# coding: utf-8

# In[2]:


from igraph import *
import xnet
import util
# import math

# In[3]:


net = xnet.xnet2igraph('citation_net_ge_1990.xnet')

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


for year in range(1990,2007):
    print('current year',year)
    colab_dict, paper_dict  = util.author_colabs_author(net,year,delta,authors)
    idxs = set()
    edges = []
    weights_basic = []
    weights_comb = []
    papers = []
    for key in colab_dict.keys():
        a1,a2 = tuple(key)
        idxs.add(a1)
        idxs.add(a2)
        edges.append((a1,a2))
        weights_basic.append(colab_dict[key])
        total_colabs = len(paper_dict[key])
        weights_comb.append(colab_dict[key]*total_colabs)
        papers.append(str(paper_dict[key]))

    idxs = list(idxs)
    print('total of authors',len(idxs),'total of colabs',len(edges))

    colab_net = Graph()
    colab_net.add_vertices(len(idxs))
    colab_net.vs['name'] = idxs
    colab_net.add_edges(edges)
    colab_net.es['weight_basic'] = weights_basic
    colab_net.es['weight_comb'] = weights_comb
    colab_net.es['papers'] = papers
    xnet.igraph2xnet(colab_net,'colabs/original/colab_'+str(year)+'_'+str(year+delta))

# In[ ]:





# In[ ]:




