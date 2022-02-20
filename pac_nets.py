import xnet
from igraph import *
import numpy as np
from itertools import combinations
from collections import defaultdict
from util import get_attr_pacs, get_pac_list
import matplotlib.pyplot as plt


def pie_dist(data, attr_pacs, pac_list, y):
    pac_count = defaultdict(lambda: 0)
    for paper in data:
        for pac in attr_pacs:
            if paper[pac][:2] in pac_list:
                pac_count[paper[pac][:2]] += 1
    x = []
    labels = []
    total = sum(list(pac_count.values()))
    others_count = 0
    for p, c in pac_count.items():
        if c > total*0.03:
            x.append(c)
            labels.append(p)
        else:
            others_count += c
    x.append(others_count)
    labels.append('Others')
    x = np.asarray(x)/total
    labels = np.asarray(labels)
    idxs = np.argsort(x)
    x = x[idxs]
    labels = labels[idxs]
    plt.pie(x, labels=labels)
    plt.title("%d (total=%d)" % (y, total))
    plt.show()


if __name__ == '__main__':
    data = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')

    attr_pacs = get_attr_pacs()
    pac_list = get_pac_list()

    delta = 2
    year_begin = 1991
    year_end = 2006

    unique_dict = defaultdict(lambda: [])

    for y in range(year_begin, year_end + 1):
        pac_control = set()

        subset = data.vs.select(year_ge=y, year_le=y + delta)

        total_papers = len(subset)
        pairs_weights = defaultdict(lambda: 0)
        for a in subset:
            pacs = []
            for pac in attr_pacs:
                if a[pac][:2] in pac_list:
                    pacs.append(a[pac][:2])
                    pac_control.add(a[pac][:2])
            #    else:
            #        if not a[pac] == 'None':
            #            print(a,a[pac])
            if len(pacs) < 2:
                continue
            pacs = sorted(pacs)
            n = len(pacs)
            combs = combinations(pacs, 2)
            for k in combs:
                pairs_weights[k] += 1 / n

        pac_net = Graph(len(pac_control))
        pac_net.vs['name'] = list(pac_control)

        edges = []
        weights = []
        for k, v in pairs_weights.items():
            edges.append((k[0], k[1]))
            weights.append(v)
        pac_net.add_edges(edges)
        pac_net.es['weight'] = weights

        print("ano:%d vertices:%d arestas:%d" % (y, pac_net.vcount(), pac_net.ecount()))
        xnet.igraph2xnet(pac_net, 'data/pacs/2lvls/pac_net_' + str(y) + '_2lvls_delta' + str(delta) + '_v3.xnet')
