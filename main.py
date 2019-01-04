import xnet
from igraph import *
import visualization
import pickle
import threading

begins = range(2000,2006)
sim_delta = 6
colab_delta = 6

# net = xnet.xnet2igraph('data/citation_network.xnet')

def plots(begin,sim_delta,colab_delta,net=None):
	# citation_net = visualization.author_cites_author(net,begin,sim_delta)
	# xnet.igraph2xnet(citation_net,'data/citation_net_authors'+str(begin)+'.xnet')

	# citation_net = xnet.xnet2igraph('data/citation_net_authors.xnet')
	# citation_net = visualization.calculate_repre(citation_net)
	# with open('temp/citation_net_'+str(begin), 'wb') as output:
	#     pickle.dump(citation_net, output, pickle.HIGHEST_PROTOCOL)

	# for r in citation_net.vs:
	# 	print(r['name'],r['repre'])

	# force = visualization.author_colabs_author(net,begin+sim_delta,colab_delta)
	# with open('temp/force_'+str(begin), 'wb') as output:
	#     pickle.dump(force, output, pickle.HIGHEST_PROTOCOL)

	# with open('citation_net','rb') as input:
	    # citation_net = pickle.load(input)

	with open('temp/force_'+str(begin),'rb') as input:
	    force = pickle.load(input)

	# valid_pairs = list(force.keys())
	# sims = visualization.calculate_sim(citation_net,valid_pairs)
	# with open('temp/sims_'+str(begin), 'wb') as output:  # Overwrites any existing file.
	#     pickle.dump(sims, output, pickle.HIGHEST_PROTOCOL)

	with open('temp/sims_'+str(begin),'rb') as input:
	    sims = pickle.load(input)

	visualization.plot_sim_vs_colab(sims,force,begin)

# net = Graph(directed=True)
# net.add_vertices(4)
# net.vs['name'] = ['p1','p2','p3','p4']
# net.vs['year'] = [2010,2009,2001,2000]
# net.vs['authors_idx'] = ['a1,a2','a1,a3,a4','a1,a2,a3','a4']
# net.add_edges([(0,2),(0,3),(1,0),(1,2),(1,3),(2,3)])

threads = []
for begin in begins:
	plots(begin,sim_delta,colab_delta)
	# t = threading.Thread(target=plots,args=(net,begin,sim_delta,colab_delta))
	# t.start()
	# threads.append(t)
# 	if len(threads) >= 5:
# 		break

# for t in threads:
# 	t.join()

print('Finished')
