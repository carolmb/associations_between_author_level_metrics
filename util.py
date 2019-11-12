from igraph import *
import xnet
import numpy as np
import time
import itertools
import json

'''
Clear database
'''
def get_net(net):

    print(net.vs.attributes())

    # titles = net.vs['title']
    # unique_titles,count_titles = np.unique(titles,return_counts=True)

    print(net.vcount(),net.ecount())

    invalid_vtxs = net.vs.select(authors_idxs_eq='')
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

    invalid_vtxs = net.vs.select(year_lt=1989)
    net.delete_vertices(invalid_vtxs)
    print(net.vcount(),net.ecount())

    '''titles = net.vs['title']
    unique_titles,count_titles = np.unique(titles,return_counts=True)
    for u,c in zip(unique_titles,count_titles):
        if c == 2:
            vs = net.vs.select(title_eq=u)
            for v in vs:
                print(v['title'],v['abstract'],v['authors_idx'])
            print()
    '''

    return net


def author_colabs_author(net,begin,delta,valid_authors):
    #begin_time = time.time()
    #print('colabs begin',begin_time,'delta',delta)

    colabs = defaultdict(lambda:0)
    papers_from_colab = defaultdict(lambda:[])
    all_authors = set()

    #print(net.vs.attributes())
    papers = net.vs.select(year_ge=begin,year_le=begin+delta)
    #papers = papers.select(year_lt=begin+delta)
    #print('total of papers:::::',len(papers))

    #middle_time = time.time()

    #print('middle delta',middle_time-begin_time)

    #print('total of papers',len(papers))
    i = 0
    for paper in papers:

        i += 1
#        if i%10000 == 0:
 #           print(i)


        authors = paper['authors_idxs'].split(',')
        authors = [a for a in authors if not a == '']
        authors = [int(a) for a in authors if int(a) in valid_authors]
        all_authors |= set(authors)
        authors = sorted(authors)
        N = len(authors)
        paper_id = paper['numeric_id']

        combinations = itertools.combinations(authors,2)

        for key in combinations:
            colabs[key] += 1/(N-1)
            papers_from_colab[key].append(paper_id)

    #end_time = time.time()
    #print('end delta',end_time-middle_time)

    print('colabs',len(colabs))
    colabs = dict(colabs)
    papers_from_colab = dict(papers_from_colab)
    return colabs,papers_from_colab,all_authors




def get_attr_pacs():
    attr_pacs = ['PACS-0','PACS-1','PACS-2','PACS-3','PACS-4']
    return attr_pacs
    
def get_pac_list():
    pac_list = ['01','02','03','04','05','06','07','11','12','13','14','21','23','24','25','26','27','28','29','31','32','33','34']
    pac_list += ['36','37','41','42','43','44','45','46','47','51','52','61','62','63','64','65','66','67','68','71','72','73','74']
    pac_list += ['75','76','77','78','79','81','82','83','84','85','87','88','89','91','92','93','94','95','96','97','98']
    pac_list = set(pac_list)
    return pac_list

def get_pacs_paper_published(paper,data):
	attr_pacs = get_attr_pacs()
	pac_list = get_pac_list()
	pacs = []
	for pac_code in attr_pacs:
		pac = paper[pac_code][:2]
		if pac[:2] in pac_list:
			pacs.append(pac)
# 	print(pacs)
	return [(pacs,len(pacs))]

def get_pacs_in(paper,data):
	pacs = []
	p_neighbors = data.neighbors(paper,mode=IN)
	for idx in p_neighbors:
		neighbor = data.vs[idx]

		pacs += get_pacs_paper_published(neighbor,data) 
	return pacs

def get_pacs_out(paper,data):
    pacs = []
    p_neighbors = data.neighbors(paper,mode=OUT)
    for idx in p_neighbors:
        neighbor = data.vs[idx]

        pacs += get_pacs_paper_published(neighbor,data) 
    return pacs

def save(data,filename):
	with open(filename, 'w') as f:
		json.dump(data, f)

def load(filename):
	data = None
	with open(filename, 'r') as f:
		data = json.load(f)
	return data

def authors_ranking(authors,N=-1):
    ranking = sorted(authors.items(),key=lambda x:x[1],reverse=True)
    if N > 0:
        return ranking[:N]
    else:
        return ranking
    
# pega a lista de PACS associados a um paper (dado o critério de get_pacs) e converte para as comunidades correspondentes
def get_pac_comm(pac_nets,paper,data,get_pacs):
    p_pacs = get_pacs(paper,data)
    p_comms = []
    for pacs,n_pacs in p_pacs:
        comms = []
        for pac in pacs:
            comms.append(pac_nets.vs.find(name=pac)['community'])
        p_comms.append((comms,n_pacs))
    return p_comms

# calcula o número de citações de um artigo e atribui o valor para cada um dos autores
def get_num_citations(data,filename):

	history = defaultdict(lambda:defaultdict(lambda:0))

	delta = 4
	year_begin = 1986
	year_end = 2006

	for i,year in enumerate(range(year_begin,year_end+1)):
		print("current year %d" % year)
		subset = data.vs.select(year_ge=year,year_le=year+delta)
		subgraph = data.subgraph(subset)
		for paper in subgraph.vs:
			authors_idxs = paper['authors_idxs'].split(',')
			num_citations = len(paper.neighbors(mode=IN))
			for author in authors_idxs:
				history[author][year+delta] += num_citations
		save(history,filename)

	return history

# normaliza os valores de um dado autor para um dado ano
'''
{
    "author_idx":
    {
        "1990":
        {
            "0":0.9, # comunidade de pacs
            "1":0.1
        }
    }

}

'''
def norm(history,y):
    for author in history.keys():
        comms_freq = history[author][y]
        total_papers = sum(comms_freq.values())
        for comm in comms_freq.keys():
            comms_freq[comm] /= total_papers
        history[author][y] = comms_freq
    return history

def get_pac_comm_freq(data,pac_nets,get_papers,delta,filename):

    history = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

    year_begin = 1986
    year_end = 2006

    for i,year in enumerate(range(year_begin,year_end+1)):
        print("current year %d" % year)
        subset = data.vs.select(year_ge=year,year_le=year+delta)
        subgraph = data.subgraph(subset)
        for paper in subgraph.vs:
            authors_idxs = paper['authors_idxs'].split(',')
            comms_by_paper = get_pac_comm(pac_nets[i],paper,subgraph,get_papers)
            for author in authors_idxs:
                for comms,n_comms in comms_by_paper:
                    for comm in comms:
                        history[author][year+delta][comm] += 1/n_comms
                    if len(comm) == 0:
                        print(authors_idxs)
        history = norm(history,year+delta)
        save(history,filename)

    return history

def min_papers_area(history,year_begin,year_end,min_papers):
    
    valid_authors_by_area = defaultdict(lambda:defaultdict(lambda:set()))
    
    for year in range(year_begin,year_end+1):
    
        comms_size = defaultdict(lambda:set())
        for author,comms in history.items():
            current_comms = comms[year]
            
            for comm,occurs in current_comms.items():
                comms_size[comm].add(author)
                if occurs >= 5:
                    valid_authors_by_area[year][comm].add(author)
        
#         print(year)
#         for key in comms_size.keys():
#             print("%s - %.2f - %d" % (key,len(valid_authors_by_area[year][key])*100/len(comms_size[key]),
#                                       len(comms_size[key])))
#         print()
    return valid_authors_by_area
        
def get_area(data,pac_nets,get_papers,delta):
    history = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

    year_begin = 1986
    year_end = 2006

    authors_total = defaultdict(lambda:defaultdict(lambda:0))
    
    for i,year in enumerate(range(year_begin,year_end+1)):
        print("current year %d" % year)
        subset = data.vs.select(year_ge=year,year_le=year+delta)
        subgraph = data.subgraph(subset)
        for paper in subgraph.vs:
            authors_idxs = paper['authors_idxs'].split(',')
            comms_by_paper = get_pac_comm(pac_nets[i],paper,subgraph,get_papers)
            for author in authors_idxs:
                authors_total[year+delta][author] += 1
                for comms,n_comms in comms_by_paper:
                    comms = set(comms) # cada comunidade deve contar apenas uma vez
                    for comm in comms:
                        history[author][year+delta][comm] += 1
                    if len(comm) == 0:
                        print(authors_idxs)
                        
    return min_papers_area(history,year_begin+delta,year_end+delta,5)
#         history = norm(history,year+delta)
#         save(history,filename)

def get_div(values):
	div = np.exp(-np.sum(values*np.log(values)))
	return div

def get_most_freq(comms_freq):
    most_freq = defaultdict(lambda:[])
    for key,value in comms_freq.items():
        most_freq[key[:1]].append(value)
    if len(most_freq) > 0:
        ket_most_freq = max(most_freq.items(), key=operator.itemgetter(1))[0]
        return ket_most_freq
    else:
        return None

def author_div(a_history):
    a_hist = dict()
    for year,comms_freq in a_history.items():
        if len(comms_freq) > 0:
            a_div = get_div(list(comms_freq.values()))
        else:
            a_div = 0
        a_hist[year] = a_div
    # plot(X,Y,filename)
    return a_hist

def authors_div(history):
    authors = dict()
    for author,a_history in history.items():
        a_div = author_div(a_history)
        authors[author] = a_div
    return authors