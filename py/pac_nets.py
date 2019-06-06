import xnet
from igraph import *
import numpy as np
from itertools import combinations
from collections import defaultdict

data = xnet.xnet2igraph('../data/citation_network_ge1990_pacs.xnet')

attr_pacs = ['PACS-0','PACS-1','PACS-2','PACS-3','PACS-4']
pac_list = ['01','02','03','04','05','06','07','11','12','13','14','21','22','23','24','25','26','27','28','29','31','32','33','34']
pac_list += ['35','36','37','41','42','43','44','45','46','47','51','52','61','62','63','64','65','66','67','68','71','72','73','74']
pac_list += ['75','76','77','78','79','81','82','83','84','85','86','87','88','89','91','92','93','94','95','96','97','98']
pac_list = set(pac_list)
delta = 3
year_begin = 1990
year_end = 2010

for y in range(year_begin,year_end+1):
    print(y)
    pac_net = Graph(len(pac_list))
    pac_net.vs['name'] = list(pac_list)

    subset = data.vs.select(year_ge=y,year_le=y+delta)
    pairs_weights = defaultdict(lambda:0)
    for a in subset:
        pacs = []
        for pac in attr_pacs:
            if a[pac][:2] in pac_list:
                pacs.append(a[pac][:2])
        #    else:
        #        if not a[pac] == 'None':
        #            print(a,a[pac])
        if len(pacs) < 2:
            continue
        pacs = sorted(pacs)
        #print(pacs)
        combs = combinations(pacs,2)

        for k in combs:
        #    print(k)
            pairs_weights[k] += 1

    edges = []
    weights = []
    for k,v in pairs_weights.items():
        edges.append((k[0],k[1]))
        weights.append(v)
    pac_net.add_edges(edges)
    pac_net.es['weight'] = weights
    xnet.igraph2xnet(pac_net,'pac_net_'+str(y)+'.xnet')

