import xnet
from igraph import *
import nltk
import re

def select_largest_component(net):
    vtotal = net.vcount()
    net.vs['idx'] = range(0,vtotal)
    print('Before:',net.vcount(),'vertices and',net.ecount(),'edges')

    und = net.as_undirected()
    print('Before:',und.vcount(),'vertices and',und.ecount(),'edges')

    decomposeds = und.decompose(mode=WEAK)
    max_vcount = 0
    max_idx = -1
    for idx,d in enumerate(decomposeds):
        try:
            v = d.vcount()
            if v > max_vcount:
                print('Current largest number of vertices',v,end='\r')
                max_vcount, max_idx = v, idx
        except:
            pass
        print()

    largest = decomposeds[max_idx]
    selected = set(largest.vs['idx'])
    largest = net.subgraph([v.index for v in net.vs if v['idx'] in selected])

    print('After:',largest.vcount(),'vertices and',largest.ecount(),'edges')
    return largest

def select_subgraph(begin,end,net):
    print('Before:',net.vcount(),'vertices and',net.ecount(),'edges')

    selected = net.vs.select(year_ge=begin,year_le=end)
    net = net.subgraph(selected)

    print('After:',net.vcount(),'vertices and',net.ecount(),'edges')
    return net

def remove_extra_attributes(net, attributes):
    for at in attributes:
    	del net.vs[at]
    return net

def rename_attributes(net,to_rename):
    for old,new in to_rename:
    	net.vs[new] = net.vs[old]
    	del net.vs[old]
    return net

grammar = "NP: {<DT>?<RB>*<JJ>*<NN><PR>*}" # source: https://www.clips.uantwerpen.be/pages/mbsp-tags
cp = nltk.RegexpParser(grammar)

def preprocess_abstract(text):
    text = text.lower()
    text = re.split(" |\(|\)|'|\[|\]|,:\.",text) # TODO: o que fazer com fórmulas, lematizar?
    text = [word for word in text if not word == '']

    if len(text) == 0:
        return ""

    # text = nltk.pos_tag(text)

    # words = [l[0] for t in cp.parse(text).subtrees() if t.label() == 'NP' for l in t.leaves()]
    text = ' '.join(text)
    # source: https://stackoverflow.com/questions/48897910/parse-nltk-tree-output-in-a-list-of-noun-phrase

    return text

def preprocess_citation_network():
    print('Loading network...')
    net = xnet.xnet2igraph(base+"wosAPSWithPACS_WithMAG_raw.xnet")
    print(net.is_directed())
    
    print('Removing attributes...')
    attributes = ['Digital Object Identifier (DOI)','Document Type',
    'Language','MAGID','PACS-1','PACS-1 Major','PACS-2',
    'PACS-2 Major','PACS-3','PACS-3 Major','PACS-4','PACS-4 Major','Page Count','hasPACS']
    net = remove_extra_attributes(net,attributes)

    print('Renaming attributes...')
    to_rename = [('Times Cited','times_cited'),('Subject Category','category'),('name','title'),('MAGAuthorsID','authors_idxs'),
    ('MAGAuthorsNames','authors_names'),('Title and Abstract','abstract'),
    ('Year Published','year'),('29-Character Source Abbreviation','source')]
    net = rename_attributes(net,to_rename)

    net = select_subgraph(1990,2010,net)
    print(net.is_directed())

    print('Title and abstract preprocess...')
    net.vs['title'] = [preprocess_abstract(t) for t in net.vs['title']]
    abstracts = [preprocess_abstract(a) for a in net.vs['abstract']]
    net.vs['abstract'] = abstracts

    xnet.igraph2xnet(net, fileName=base+'citation_network_ge1990.xnet')

'''
original vertices attributes:
	['name',
	'29-Character Source Abbreviation',
	'Digital Object Identifier (DOI)',
	'Document Type',
	'JCR Category',
	'Language',
	'MAGAuthorsID',
	'MAGAuthorsNames',
	'MAGID',
	'PACS-0',
	'PACS-0 Major',
	'PACS-1',
	'PACS-1 Major',
	'PACS-2',
	'PACS-2 Major',
	'PACS-3',
	'PACS-3 Major',
	'PACS-4',
	'PACS-4 Major',
	'Page Count',
	'Subject Category',
	'Times Cited',
	'Title and Abstract',
	'Year Published',
	'hasAbstract',
	'hasPACS',
	'id']
'''
base = "../data/"

preprocess_citation_network()

