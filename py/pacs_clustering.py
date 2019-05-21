from igraph import *
import numpy as np
import xnet
from collections import defaultdict
import glob
import ast

files = 'colabs/basic_colab_cut/*0.5*selected*basic4.xnet'
bla = '../data/citation_net_ge_1990.xnet'

files = glob.glob(files)
files = sorted(files)

data = xnet.xnet2igraph('../data/citation_network_ge1990.xnet')
data.vs['numeric_id'] = range(data.vcount())
print('0.5 preserved')

def get_top3(paper_infos,param,sep):
    attr_by_param = paper_infos[param]
    attr = []
    for a in attr_by_param:
        a = a.split(sep)
        a = [b.strip() for b in a if not b == 'None' and not b == '']
        attr += a

    attr = sorted(list(set(attr)))
    out = ','.join(attr)
    return out

def get_papers(colab_net,author_vtx):
    edges = g.incident(author)
    papers = set()
    for e in edges:
        papers |= set(ast.literal_eval(g.es[e]['papers']))
    
def to_str(pac_list):
    pac_str = ""
    for paper_pacs in pac_list:
        for pac in paper_pacs:
            pac = pac[:2]
            pac_str += " " + pac
    return pac_str

def authors2vec(colab_net,citation_net):
    authors = colab_net.vs
    vecs = []

    for author in authors:
        if i%10000 == 0:
            print(i)
        paper_idxs = get_papers(colab_net,author)
        papers = citation_net.select(numeric_id_in=paper_idxs)
        pacs = papers['PACS-0']
        vec = to_str(pacs)
        vecs.append(vec)

    authors['pac_vec'] = vecs
    return vecs