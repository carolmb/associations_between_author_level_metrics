import xnet
from igraph import *
import util
import numpy as np
import glob
import ast
import matplotlib.pyplot as plt
import time
import itertools

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
	papers = [np.asarray(ast.literal_eval(ps)) for ps in papers]

	papers_len = [len(ps) for ps in papers]
	return get_freq(papers_len)

def get_papers_citation_count(net):
    papers = net.es['papers']
    papers = [ast.literal_eval(ps) for ps in papers]

    papers_ids = set(itertools.chain.from_iterable(papers))
    papers_citation_count = citation_net.vs.select(numeric_id_in=papers_ids)['times_cited']

    return get_freq(papers_citation_count)

def time_series(filenames_seq,years):
    prefix = 'pdfs/'
    means = []

    for filenames in filenames_seq:
        means.append([])
        for f in filenames:
            print(f)
            t0 = time.time()
            net = xnet.xnet2igraph(f)
            t1 = time.time()
            print('reading graph from file',t1-t0)
            _,_,citation_count = get_papers_citation_count(net)
            t2 = time.time()
            print('papers citation count',t2-t1)
            means[-1].append(np.mean(citation_count))
            print()
    plot_line(years,means,'citation mean',prefix + 'citation_mean.pdf')

def plot_by_year(filenames):
	prefix = 'pdfs/'
	for filenames in filenames_seq:
		for f in filenames:
			net = xnet.xnet2igraph(f)
			f = f.split('/')[-1][:-5]

			xs,ys,_ = get_number_of_papers_by_colab(net)
			name = prefix + f + '_number_of_papers_by_colab_freq.pdf'
			plot_frequency(xs,ys,'number of papers frequency',name)

			xs,ys,_ = get_papers_citation_count(net)
			name = prefix + f + '_papers_citation_count_freq.pdf'
			plot_frequency(xs,ys,'citation count frequency',name,True)

citation_net_name = 'citation_net_ge_1990.xnet'
# run just one time
# citation_net = util.get_net()
# xnet.igraph2xnet(citation_net,citation_net_name)
# del citation_net

citation_net = xnet.xnet2igraph(citation_net_name)
vcount = citation_net.vcount()
citation_net.vs['numeric_id'] = range(vcount)

filenames_seq = []
headers = ['colabs/basic_colab_cut/*deleted_basic.xnet','colabs/basic_colab_cut/*selected_basic.xnet']

for header in headers:
	filenames = glob.glob(header)
	filenames = sorted(filenames)
	filenames_seq.append(filenames)

years = list(range(1990,2011))
print(years)
print(filenames_seq)
time_series(filenames_seq,years)
plot_by_year(filenames_seq)

filenames_seq = []
headers = ['colabs/comb_colab_cut/*deleted*.xnet','colabs/comb_colab_cut/*selected*.xnet']

for header in headers:
	filenames = glob.glob(header)
	filenames = sorted(filenames)
	filenames_seq.append(filenames)

years = list(range(1990,2011))
print(years)
print(filenames_seq)
time_series(filenames_seq,years)
plot_by_year(filenames_seq)

filenames_seq = []
headers = ['colabs/comb_colab_log_cut/*deleted*.xnet','colabs/comb_colab_log_cut/*selected*.xnet']

for header in headers:
	filenames = glob.glob(header)
	filenames = sorted(filenames)
	filenames_seq.append(filenames)

years = list(range(1990,2011))
print(years)
print(filenames_seq)
time_series(filenames_seq,years)
plot_by_year(filenames_seq)