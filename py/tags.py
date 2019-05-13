from igraph import *
import numpy as np
import xnet
from collections import defaultdict
import glob
import ast

files = 'colabs/basic_colab_cut/*selected*attrs.xnet'
bla = '../data/citation_net_ge_1990.xnet'

files = glob.glob(files)

# data = xnet.xnet2igraph('../data/citation_network_ge1990.xnet')
# data.vs['numeric_id'] = range(data.vcount())

def get_top3(paper_infos,param,sep):
    attr_by_param = paper_infos[param]
    attr = []
    for a in attr_by_param:
        a = a.split(sep)
        a = [b.strip() for b in a]
        attr += a

    attr = sorted(list(set(attr)))
    out = ','.join(attr)
    return out

begin = 1990
delta = 3
for file in files:
    # print(begin)
    # subdata = data.vs.select(year_ge=begin,year_le=begin+delta)
    g = xnet.xnet2igraph(file)
    print(g.es.attributes())
    g.es['weight'] = g.es['weight_basic']
    del g.es['weight_basic']
    # i = 0
    # print('total of authors',g.vcount())
    # for author in g.vs:
    #     if i%1000 == 0:
    #         print(i)
    #     edges = g.incident(author)
    #     papers = set()
    #     for e in edges:
    #         papers |= set(ast.literal_eval(g.es[e]['papers']))
    #     paper_infos = subdata.select(numeric_id_in=papers)
    #     author['PACS-0'] = get_top3(paper_infos,'PACS-0',',')
    #     author['category'] = get_top3(paper_infos,'category',';')
    #     i += 1
    xnet.igraph2xnet(g,file[:-5]+'attrs.xnet')
    begin += 1