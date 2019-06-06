from igraph import *
import numpy as np
import xnet
from collections import defaultdict
import glob
import pickle
import ast
from sklearn.utils.random import sample_without_replacement
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


files = 'colabs/wbasic/*0.5*selected_wb.xnet'

files = glob.glob(files)
files = sorted(files)

data = xnet.xnet2igraph('../data/citation_network_ge1990_pacs.xnet')

def get_papers(colab_net,author_vtx):
    edges = colab_net.incident(author_vtx)
    papers = set()
    for e in edges:
        papers |= set(ast.literal_eval(colab_net.es[e]['papers']))
    return papers

def to_str(pac_list):
    pac_str = "none"
    for pac in pac_list:
        if pac == 'None':
            continue
        pac = pac[:2]
        pac_str += " " + pac
        #if "," in pac_list or ";" in pac_list:
            #print("ERRO, PRECISA APLICAR SPLIT")

    #print(pac_str)
    return pac_str

def authors2str(colab_net,citation_net):
    authors = colab_net.vs
    vecs = []

    i = 0
    for author in authors:
        i+=1
        if i%10000 == 0:
            print(i)

        paper_idxs = get_papers(colab_net,author)
        papers = citation_net.vs.select(numeric_id_in=paper_idxs)
        if not len(paper_idxs) == len(papers):
            print("DEU RUIM")
        pacs = papers['PACS-0'] + papers['PACS-1'] + papers['PACS-2'] + papers['PACS-3'] + papers['PACS-4']
        vec = to_str(pacs)
        vecs.append(vec)

    return vecs

def authors2vec(vecs):
    vocab = ['01','02','03','04','05','06','07','11','12','13','14','21','22','23','24','25','26','27','28','29','31','32','33','34']
    vocab += ['35','36','37','41','42','43','44','45','46','47','51','52','61','62','63','64','65','66','67','68','71','72','73','74']
    vocab += ['75','76','77','78','79','81','82','83','84','85','86','87','88','89','91','92','93','94','95','96','97','98','none']
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(vecs)
    #print(vectorizer.vocabulary_)
    return X.toarray()

def clustering_by_pac(colab_net,citation_net,year):
    vecs = authors2str(colab_net,citation_net)
    bow = authors2vec(vecs)
    for i,line in enumerate(bow):
        colab_net.vs[i]['bow'] = str(line.tolist())
    #for line in bow[:100]:
    #    print(line.tolist())

    clustering = AgglomerativeClustering(n_clusters=5,affinity='cosine',linkage='average')
    fited_clustering = clustering.fit(bow)
    labels = fited_clustering.labels_
    labels = [str(l) for l in labels]
    colab_net.vs['pac_hier'] = labels
    print('hierarquico',np.unique(labels,return_counts=True))

    clustering = KMeans(n_clusters=5,random_state=0)
    fited_clustering = clustering.fit(bow)
    labels = fited_clustering.labels_
    labels = [str(l) for l in labels]
    print('kmeans',np.unique(labels,return_counts=True))

    colab_net.vs['pac_kmeans'] = labels

    #pca = PCA(n_components=3)
    #bow_3d = pca.fit_transform(bow)
    #print(bow_3d)
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')

    #color = {'0':'red','1':'blue','2':'green','3':'black','4':'orange','5':'yellow','6':'pink','7':'purple','8':'magenta','9':'cyan'}
    #colors = [color[l] for l in labels]
    #ax.scatter(bow_3d[:,0],bow_3d[:,1],bow_3d[:,2],c=colors,alpha=0.5)
    #plt.savefig('test_kmeans_'+str(year)+'.pdf',format='pdf')
    return colab_net

def clustering_by_pac_cluster(colab_net,citation_net,pac_net):
    vecs = authors2str(colab_net,citation_net)
    n = len(set(pac_net.vs['community']))
    reduced_vecs = []
    names = set(pac_net.vs['name'])
    for vec in vecs:
        v = [0]*(n+1)
        v[-1] = 1
        vec = vec.split(' ')
        for pac in vec:
            if pac == 'none' or pac == '99' or not pac in names:
                continue
                # print(pac)
            k = int(pac_net.vs.find(name=pac)['community'])-1
            v[k] += 1
        print(v)
        reduced_vecs.append(v)
    colab_net.vs['reduced_bow'] = [str(v) for v in reduced_vecs]
    
    clustering = AgglomerativeClustering(n_clusters=30,affinity='cosine',linkage='average')
    fited_clustering = clustering.fit(reduced_vecs)
    labels = fited_clustering.labels_
    labels = [str(l) for l in labels]
    colab_net.vs['pac_hier'] = labels
    print('hierarquico',np.unique(labels,return_counts=True))

def pacs_by_intervals():
    year = 1990
    for f in files:
        print(f)
        colab_net = xnet.xnet2igraph(f)
        pac_net = xnet.xnet2igraph('pacs/pac_net_'+str(year)+'_leidenalg.xnet')

        vtx2del = [vtx for vtx in colab_net.vs if colab_net.degree(vtx) == 0]
        colab_net.delete_vertices(vtx2del)

        clustering_by_pac_cluster(colab_net,data,pac_net)
        xnet.igraph2xnet(colab_net,f[:-5]+'_pac5_cluster.xnet')
        year += 1

pacs_by_intervals()
# attr_pacs = ['PACS-0','PACS-1','PACS-2','PACS-3','PACS-4']
# pacs = set()
# for y in range(1995,2010):
#     for v in data.vs.select(year_ge=y,year_le=y+3):
#         for pac in attr_pacs:
#             if v[pac][:2] == 'PA':
#                 print(v[pac],end='  ')
#             pacs.add(v[pac][:2])
#     print()
#     print(y)
#     print(sorted(list(pacs)))
#     break
