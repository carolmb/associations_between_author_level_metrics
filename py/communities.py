import xnet
import subprocess
import igraph
import leidenalg
import glob

def xnet2net(g,outputname):
	output = open(outputname,'w')
	vcount = g.vcount()
	ecount = g.ecount()
	output.write('*Vertices ' + str(vcount) + '\n')

	vtxs = g.vs
	for v in vtxs:
		output.write(str(v.index+1) + ' "' + v['name'] + '"\n')

	output.write('*Edges ' + str(ecount) + '\n')

	edges = g.es
	for e in edges:
		output.write(str(e.tuple[0]+1) + ' ' + str(e.tuple[1]+1) + ' ' + str(e['weight']) + '\n')

def get_communities_by_level(level,filename):
	finput = open(filename,'r').read().split('\n')[2:-1]
	communities = []
	for line in finput:
		line = line.split(' ')
		node_idx = int(line[-1])-1

		hierar = line[0].split(':')

		idx = min(level,len(hierar))
		c = ':'.join(hierar[:idx])

		communities.append((node_idx,c))
	return communities

def identify_communities_infomap(net,lvl=1):
	temp = 'temp'
	path = 'output/'

	xnet2net(net,temp+'.net')
	subprocess.call(["./Infomap/Infomap",temp+'.net',path,'--tree','-N1000'])
	communities = get_communities_by_level(lvl,path+temp+'.tree')

	net_copy = net.copy()
	for node_idx,c in communities:
		net_copy.vs[node_idx]['community'] = c

	subprocess.call(["rm",temp+'.net'])
	subprocess.call(["rm",path+temp+'.tree'])
	return net_copy

def get_largest_component(g):
		components = g.components()
		giant = components.giant()
		return giant

def identify_communities_multilevel(net,lvl):
	giant = get_largest_component(net)
	comm_level = giant.community_multilevel(weights='weight',return_levels=True)
	t = len(comm_level)
	comms = comm_level[min(lvl,t)]
	comm_list = comms.subgraphs() # communities in current level
	print('Level',min(lvl,t),'Number of communities identified:',len(comm_list))

	net_copy = net.copy()
	net_copy.vs['community'] = "-1"
	for idx,c_graph in enumerate(comm_list):
		for v1 in c_graph.vs:
			v2 = net_copy.vs.find(name=v1['name'])
			v2['community'] = str(idx+1)
	return net_copy

def identify_communities_leidenalg(net):
    giant = net #giant = get_largest_component(net)
    comms = leidenalg.find_partition(giant, leidenalg.ModularityVertexPartition)
    comm_list = comms.subgraphs() # communities in current level
    print('Number of communities identified:',len(comm_list))
    net_copy = net.copy()
    net_copy.vs['community'] = "-1"
    for idx,comm in enumerate(comm_list):
        for v1 in comm.vs:
            v2 = net_copy.vs.find(name=v1['name'])
            v2['community'] = str(idx+1)
    return net_copy

filenames = glob.glob("pacs/*.xnet")
filenames = sorted(filenames)

print(filenames,'aaaa')

graphs = []
for filename in filenames:
	print(filename)
	net = xnet.xnet2igraph(filename)
	net = identify_communities_infomap(net)

	output = filename[:-5] + '_infomap.xnet'
	xnet.igraph2xnet(net,output)
