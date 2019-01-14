import xnet
from igraph import *
import visualization
import pickle
import threading
import numpy as np

begins = range(2000,2006)
sim_delta = 10
colab_delta = 10

net = xnet.xnet2igraph('data/citation_network.xnet')

titles = net.vs['title']
unique_titles,count_titles = np.unique(titles,return_counts=True)

idxs = np.argsort(count_titles)
# print(count_titles[idxs[-100:]])
print(net.vcount(),net.ecount())
invalid_vtxs = net.vs.select(authors_idx_eq='')
net.delete_vertices(invalid_vtxs)
print(net.vcount(),net.ecount())

invalid_vtxs = net.vs.select(abstract_eq='')
net.delete_vertices(invalid_vtxs)
print(net.vcount(),net.ecount())

invalid_vtxs = net.vs.select(title_eq='')
net.delete_vertices(invalid_vtxs)
print(net.vcount(),net.ecount())

invalid_vtxs = net.vs.select(title_eq='CORRECTION')
net.delete_vertices(invalid_vtxs)
print(net.vcount(),net.ecount())

def plot(begin,sim_delta,colab_delta,net=None):
    '''
    generates the citation (a1 cites a2) net considering an interval of time
    '''	
    citation_net,valid_authors = visualization.author_cites_author(net,begin,sim_delta)
    
    citation_net = visualization.repre_attribute(citation_net)
    # with open('temp/citation_net_'+str(begin), 'wb') as output:
    #     pickle.dump(citation_net, output, pickle.HIGHEST_PROTOCOL)

    with open('temp/citation_net_'+str(begin),'rb') as input:
    	citation_net = pickle.load(input)

    '''
    coauthorship pairs considering the original citation network, an interval of time and the valid authors
    '''
    force = visualization.author_colabs_author(net,begin+sim_delta,colab_delta,valid_authors)
    with open('temp/force_'+str(begin), 'wb') as output:
        pickle.dump(force, output, pickle.HIGHEST_PROTOCOL)

    # with open('temp/force_'+str(begin),'rb') as input:
    #     force = pickle.load(input)

    coauthorship_pairs = list(force.keys())
    print('coauthorship_pairs',coauthorship_pairs)
    sims = visualization.calculate_sim(citation_net,coauthorship_pairs)
    with open('temp/sims_'+str(begin), 'wb') as output:
        pickle.dump(sims, output, pickle.HIGHEST_PROTOCOL)
    
    visualization.plot_sim_vs_colab(sims,force,begin)

# net = Graph(directed=True)
# net.add_vertices(4)
# net.vs['name'] = ['p1','p2','p3','p4']
# net.vs['year'] = [2010,2009,2001,2000]
# net.vs['authors_idx'] = ['a1,a2','a1,a3,a4','a1,a2,a3','a4']
# net.add_edges([(0,2),(0,3),(1,0),(1,2),(1,3),(2,3)])

for begin in begins:
    plot(begin,sim_delta,colab_delta,net)
    break
print('Finished')
