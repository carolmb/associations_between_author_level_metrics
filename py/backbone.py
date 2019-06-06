#!/usr/bin/env python
# coding: utf-8

# In[2]:


import xnet
from igraph import *
from scipy import integrate
import glob
import numpy as np
import pickle
import time

# source: https://github.com/aekpalakorn/python-backbone-network/blob/master/backbone.py
def disparity_filter(g,weight_key):
    total_vtx = g.vcount()
    g.es['alpha_ij'] = 1

    for v in range(total_vtx):
        edges = g.incident(v)

        k = len(edges)
        if k > 1:
            sum_w = sum([g.es[e][weight_key] for e in edges])
            for e in edges:
                w = g.es[e][weight_key]
                p_ij = w/sum_w
                alpha_ij = 1 - (k-1) * integrate.quad(lambda x: (1-x)**(k-2), 0, p_ij)[0]
                g.es[e]['alpha_ij'] = min(alpha_ij,g.es[e]['alpha_ij'])

def alpha_cut(alpha,g):
    g_copy1 = g.copy()
    to_delete = g_copy1.es.select(alpha_ij_ge=alpha)
    g_copy1.delete_edges(to_delete)

    #g_copy2 = g.copy()
    to_delete = g.es.select(alpha_ij_lt=alpha)
    #g_copy2.delete_edges(to_delete)
    return g_copy1,to_delete

def get_largest_component_size(g):
    components = g.components()
    giant = components.giant()
    return giant.vcount()

def get_best_cut(net,preserve_percent,a_min,a_max):
    t0 = time.time()
    error = 0.015
    largest_size = get_largest_component_size(net)

    i = 0

    a_min_perc = get_largest_component_size(alpha_cut(a_min,net)[0])/largest_size
    a_max_perc = get_largest_component_size(alpha_cut(a_max,net)[0])/largest_size

    alpha = 0.0

    t1 = time.time()
    print('while begin',t1-t0)
    while True:
        if i > 50:
            print('i > 100')
            cuted_net,deleted_edges = alpha_cut(a_min,net)
            preserved_size = get_largest_component_size(cuted_net)
            current_percent = preserved_size/largest_size
            print('error: current percent:',current_percent)
            print()
            net.delete_edges(deleted_edges)
            return cuted_net,net

        t2 = time.time()
        alpha = (a_min+a_max)/2

        cuted_net,deleted_edges = alpha_cut(alpha,net)

        preserved_size = get_largest_component_size(cuted_net)

        current_percent = preserved_size/largest_size

        current_erro = current_percent-preserve_percent

        if abs(current_erro) < error:
            print('total iterations to find the graph',i)
            print('alpha %.2f; preserved %.2f' % (alpha,current_percent))
            print()
            net.delete_edges(deleted_edges)
            return cuted_net,net

        if (a_min_perc-preserve_percent)*(current_percent-preserve_percent) > 0:
            a_min = alpha
            a_min_perc = current_percent
        else:
            a_max = alpha
            a_max_perc = current_percent

        i += 1
        t3 = time.time()
        print('iteration time seconds',t3-t2)
# In[5]:


def apply_backbone(net,a_min,a_max,preserve=0.6,weight_key='weight_basic'):
    t0 = time.time()
    print('apply_backbone begin')
    giant = net.components().giant()
    disparity_filter(giant,weight_key)
    t1 = time.time()
    print('disparity_filter end',t1-t0)
    return get_best_cut(giant,preserve,a_min,a_max)


# In[8]:

def generate_nets(filenames,prefix,sufix,weight_key,preserve):
    a_min = 0.0001
    a_max = 1

    for filename in filenames:
        print(filename)
        net = xnet.xnet2igraph(filename)
        if net.ecount() > 0:
            # print(net.ecount())
            net_selected,net_deleted = apply_backbone(net,a_min,a_max,preserve,weight_key)

            print('saving the graphs')
            t0 = time.time()
            output = prefix + filename.split('/')[-1] + '_' + str(preserve) + '_selected'+ sufix
            xnet.igraph2xnet(net_selected,output)
            output = prefix + filename.split('/')[-1] + '_' + str(preserve) + '_deleted' + sufix
            xnet.igraph2xnet(net_deleted,output)
            t1 = time.time()
            print('time saving graphs',t1-t0)

filenames = glob.glob('colabs/original/*')
filenames = sorted(filenames)
print(filenames)

'''
preserve = 0.5
weight_key = 'weight_basic'
prefix = 'colabs/wbasic/'
sufix = '_wb.xnet'
generate_nets(filenames,prefix,sufix,weight_key,preserve)
'''
preserve = 0.5
weight_key = 'weight'
prefix = 'colabs/wcomb/'
sufix = '_wc.xnet'
generate_nets(filenames,prefix,sufix,weight_key,preserve)
'''
preserve = 0.4
weight_key = 'weight_comb_log'
prefix = 'colabs/comb_colab_log_cut/'
sufix = '_comb_log4.xnet'
generate_nets(filenames,prefix,sufix,weight_key,preserve)
'''
