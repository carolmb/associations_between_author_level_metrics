import xnet
from igraph import *
import util
import numpy as np
import glob
import ast
import matplotlib.pyplot as plt

def plot_frequency(xs,ys,xlabel,name,islog=False):
	plt.figure(figsize=(4,3))
	plt.bar(xs, ys)
	plt.xlabel(xlabel)
	plt.ylabel("frequency")
	if islog:
		plt.yscale("log")
	plt.savefig(name,format='pdf',bbox_inches="tight")
	plt.close()

def plot_line(xs,all_ys,ylabel,name):
	plt.figure(figsize=(12,3))
	for ys in all_ys:
		plt.plot(xs,ys,linestyle='-',linewidth=2,markersize=2,alpha=0.8)
	plt.xlabel("year")
	plt.ylabel(ylabel)
	plt.savefig(name,format='pdf',bbox_inches="tight")
	plt.close()

def get_freq(info):
	xs,ys = np.unique(info,return_counts=True)
	return xs,ys,info

def get_number_of_papers_by_colab(net):
	papers = net.es['papers']
	papers = [ast.literal_eval(ps) for ps in papers]
	
	papers_len = [len(ps) for ps in papers]
	return get_freq(papers_len)

def get_papers_citation_count(net):
	papers = net.es['papers']
	papers = [ast.literal_eval(ps) for ps in papers]
	
	papers_ids = set([p for ps in papers for p in ps])
	papers_citation_count = citation_net.vs.select(id_in=papers_ids)['times_cited']

	return get_freq(papers_citation_count)

def time_series(filenames_seq,years):
	prefix = 'pdfs/'
	means = []
	
	for filenames in filenames_seq:
		means.append([])
		for f in filenames:
			net = xnet.xnet2igraph(f)
			_,_,citation_count = get_papers_citation_count(net)
			means[-1].append(np.mean(citation_count))
			del net
	plot_line(years,means,'citation mean',prefix + 'citation_mean.pdf')

def plot_by_year(filenames):
	prefix = 'pdfs/'
	for f in filenames:
		net = xnet.xnet2igraph(f)
		f = f.split('/')[-1][:-5]

		xs,ys,_ = get_number_of_papers_by_colab(net)
		name = prefix + f + '_number_of_papers.pdf'
		plot_frequency(xs,ys,'number of papers',name)

		xs,ys,citation_count = get_papers_citation_count(net)
		name = prefix + f + '_papers_citation_count.pdf'
		plot_frequency(xs,ys,'citation count',name,True)

citation_net_name = 'citation_net_ge_1990.xnet'
# run just one time
# citation_net = util.get_net()
# xnet.igraph2xnet(citation_net,citation_net_name)
# del citation_net

citation_net = xnet.xnet2igraph(citation_net_name)

filenames_seq = []
headers = ['colabs/basic_colab_cut/*deleted_basic.xnet','colabs/basic_colab_cut/*selected_basic.xnet']

for header in headers:
	filenames = glob.glob(header)
	filenames = sorted(filenames)
	filenames_seq.append(filenames)

years = list(range(1990,2007))
print(filenames_seq)
time_series(filenames_seq,years)