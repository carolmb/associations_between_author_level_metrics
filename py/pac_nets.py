import xnet
from igraph import *
import numpy as np
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt

data = xnet.xnet2igraph('../data/citation_network_ge1990_pacs.xnet')

attr_pacs = ['PACS-0','PACS-1','PACS-2','PACS-3','PACS-4']
pac_list = ['01','02','03','04','05','06','07','11','12','13','14','21','23','24','25','26','27','28','29','31','32','33','34']
pac_list += ['36','37','41','42','43','44','45','46','47','51','52','61','62','63','64','65','66','67','68','71','72','73','74']
pac_list += ['75','76','77','78','79','81','82','83','84','85','87','88','89','91','92','93','94','95','96','97','98']
pac_list = set(pac_list)
delta = 3
year_begin = 1990
year_end = 2010
unique_dict = defaultdict(lambda:[])
for y in range(year_begin,year_end+1):
    pac_control = set()
    print(y)
    
    subset = data.vs.select(year_ge=y,year_le=y+delta)
    total_papers = len(subset)
    pairs_weights = defaultdict(lambda:0)
    for a in subset:
        pacs = []
        for pac in attr_pacs:
            if a[pac][:2] in pac_list:
                pacs.append(a[pac][:4])
                pac_control.add(a[pac][:4])
        #    else:
        #        if not a[pac] == 'None':
        #            print(a,a[pac])
        if len(pacs) <= 2:
            continue
        pacs = sorted(pacs)
        #print(pacs)
        combs = combinations(pacs,2)
        n1 = len(pacs)
        for k in combs:
        #    print(k)
           pairs_weights[k] += 1/n1
    
    pac_net = Graph(len(pac_control))
    pac_net.vs['name'] = list(pac_control)

    edges = []
    weights = []
    for k,v in pairs_weights.items():
        edges.append((k[0],k[1]))
        weights.append(v)
    pac_net.add_edges(edges)
    pac_net.es['weight'] = weights
    xnet.igraph2xnet(pac_net,'pacs/pac_net_'+str(y)+'_3lvls.xnet')

#     us,cs = np.unique(pac_control,return_counts=True)
#     # todo: usar lista de pacs
#     for u,c in zip(us,cs):
#         # print(u,c)
#         unique_dict[u].append(c)
#     for u in pac_list:
#         if not u in us:
#             unique_dict[u].append(0)
# print(unique_dict)
# unique_freqs = []
# labels = []
# total = []

# for u,c in unique_dict.items():
#     if sum(c) > 500:
#         unique_freqs.append(np.asarray(c))
#         labels.append(u)
#         total.append((sum(c),u))

# unique_freqs = np.asarray(unique_freqs)

# print('unique_freqs',unique_freqs.shape)

# labels = np.asarray(labels)
# idxs = np.argsort(total,axis=0)[:,0]

# print('idxs',idxs)

# unique_freqs = unique_freqs[idxs]
# labels = labels[idxs]
# print(unique_freqs.shape)

# ys = np.cumsum(unique_freqs,axis=0)
# print('ys',ys)

# xs = [i for i in range(year_begin,year_end+1)]
# fig = plt.figure(figsize=(12,4))
# ax1 = fig.add_subplot(111)
# for l,y in zip(reversed(labels),reversed(ys)):
#     #print(y)
#     #print(y.shape)
#     ax1.fill_between(xs,y,label=l)
# plt.legend()
# plt.tight_layout()
# plt.show()
# plt.savefig('teste.pdf',format='pdf',bbox_inches='tight')
