import numpy as np
from igraph import *
import xnet
import glob
# import matplotlib.pyplot as plt
import json
import math
import concurrent.futures

np.set_printoptions(precision=6)

headers = ['colabs/basic_colab_cut/*0.5*selected_basic4.xnet']

filenames_seq = []

for header in headers:
	filenames = glob.glob(header)
	filenames = sorted(filenames)
	filenames_seq.append(filenames)

for i,filenames in enumerate(filenames_seq):
    current_year = 1990
    all_years_metrics = dict()
    all_years_metrics['infos'] = []
    info = ''
    for filename in filenames:

        year = current_year
        # print(year,'(',filename,')')
        current_year += 1

        graph = xnet.xnet2igraph(filename)
        # graph.es['weight'] = graph.es['weight_basic']
        # del graph.es['weight_basic']
        print(year,'vertices: ' + str(graph.vcount())+'vertices (giant): ' + str(graph.components().giant().vcount()) + ' edges: ' + str(graph.ecount()))
        info += 'vertices ' + str(graph.vcount()) + ' edges: ' + str(graph.ecount()) + '\n'
        # xnet.igraph2xnet(graph,filename)

    colab_type = headers[i].split('/')[1]
    edge_type = headers[i].split('/')[2][1:-5]
    base = colab_type + '_' + edge_type

    output = open("vcount_ecount__0.5_"+base+"4.json",'w')
    output.write(info)
    output.close()

