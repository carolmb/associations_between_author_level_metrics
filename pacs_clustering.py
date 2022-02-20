import ast
import glob
import xnet
import numpy as np
import matplotlib.pyplot as plt

from igraph import *
from textwrap import wrap
from sklearn.cluster import KMeans
from util import get_attr_pacs, get_pac_list
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer

# data = xnet.xnet2igraph('../data/citation_network_ge1990_pacs.xnet')


def get_papers(colab_net, author_vtx):
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
        # if "," in pac_list or ";" in pac_list:
        # print("ERRO, PRECISA APLICAR SPLIT")

    # print(pac_str)
    return pac_str


def authors2str(colab_net, citation_net):
    authors = colab_net.vs
    vecs = []

    i = 0
    for author in authors:
        i += 1
        if i % 10000 == 0:
            print(i)

        paper_idxs = get_papers(colab_net, author)
        papers = citation_net.vs.select(numeric_id_in=paper_idxs)
        if not len(paper_idxs) == len(papers):
            print("DEU RUIM")
        pacs = papers['PACS-0'] + papers['PACS-1'] + papers['PACS-2'] + papers['PACS-3'] + papers['PACS-4']
        vec = to_str(pacs)
        vecs.append(vec)

    return vecs


def authors2vec(vecs):
    vocab = ['01', '02', '03', '04', '05', '06', '07', '11', '12', '13', '14', '21', '22', '23', '24', '25', '26', '27',
             '28', '29', '31', '32', '33', '34']
    vocab += ['35', '36', '37', '41', '42', '43', '44', '45', '46', '47', '51', '52', '61', '62', '63', '64', '65',
              '66', '67', '68', '71', '72', '73', '74']
    vocab += ['75', '76', '77', '78', '79', '81', '82', '83', '84', '85', '86', '87', '88', '89', '91', '92', '93',
              '94', '95', '96', '97', '98', 'none']
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(vecs)
    # print(vectorizer.vocabulary_)
    return X.toarray()


def clustering_by_pac(colab_net, citation_net, year):
    vecs = authors2str(colab_net, citation_net)
    bow = authors2vec(vecs)
    for i, line in enumerate(bow):
        colab_net.vs[i]['bow'] = str(line.tolist())
    # for line in bow[:100]:
    #    print(line.tolist())

    clustering = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage='average')
    fited_clustering = clustering.fit(bow)
    labels = fited_clustering.labels_
    labels = [str(l) for l in labels]
    colab_net.vs['pac_hier'] = labels
    print('hierarquico', np.unique(labels, return_counts=True))

    clustering = KMeans(n_clusters=5, random_state=0)
    fited_clustering = clustering.fit(bow)
    labels = fited_clustering.labels_
    labels = [str(l) for l in labels]
    print('kmeans', np.unique(labels, return_counts=True))

    colab_net.vs['pac_kmeans'] = labels

    # pca = PCA(n_components=3)
    # bow_3d = pca.fit_transform(bow)
    # print(bow_3d)
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')

    # color = {'0':'red','1':'blue','2':'green','3':'black','4':'orange','5':'yellow','6':'pink','7':'purple','8':'magenta','9':'cyan'}
    # colors = [color[l] for l in labels]
    # ax.scatter(bow_3d[:,0],bow_3d[:,1],bow_3d[:,2],c=colors,alpha=0.5)
    # plt.savefig('test_kmeans_'+str(year)+'.pdf',format='pdf')
    return colab_net


def get_largest_component(g):
    components = g.components()
    giant = components.giant()
    return giant


def identify_communities_multilevel(net, lvl):
    giant = get_largest_component(net)
    comm_level = giant.community_multilevel(weights='weight', return_levels=True)
    t = len(comm_level) - 1
    comms = comm_level[min(lvl, t)]
    comm_list = comms.subgraphs()  # communities in current level
    print('Level', min(lvl, t), 'Number of communities identified:', len(comm_list))

    net_copy = net.copy()
    net_copy.vs['community'] = "-1"
    for idx, c_graph in enumerate(comm_list):
        for v1 in c_graph.vs:
            v2 = net_copy.vs.find(name=v1['name'])
            v2['community'] = str(idx + 1)
    return net_copy

def pac_nets_comm(files):
    for f in files:
        pac_net = xnet.xnet2igraph(f)

        pac_net = identify_communities_multilevel(pac_net, 0)
        # xnet.igraph2xnet(pac_net, f[:-5]+'_multilevel2.xnet')


def authors_to_str():
    year = 1990
    for f in files:
        print(f)
        colab_net = xnet.xnet2igraph(f)

        vtx2del = [vtx for vtx in colab_net.vs if colab_net.degree(vtx) == 0]
        colab_net.delete_vertices(vtx2del)

        vecs = authors2str(colab_net, data)
        for vtx, vec in zip(colab_net.vs, vecs):
            vtx['pac_list'] = vec

        xnet.igraph2xnet(colab_net, f[:-5] + '_with_author_pac_list.xnet')
        year += 1


if __name__ == '__main__':
    code2name = {'01': "Communication, education, history, and philosophy",
    '02': "Mathematical methods in physics",
    '03': "Quantum mechanics, fiel theories, and special relativity",
    '04': "General relativity and gravitation",
    '05': "Statistical physics, thermodynamics, and nonlinear dynamical systems",
    '11': "General theory of fields and particles",
    '12': "Specific theories and interaction models; particle systematics",
    '13': "Specific reactions and phenomenology",
    '14': "Properties of specific particles",
    '21': 'Nuclear structure',
    '24': 'Nuclear reactions: general',
    '25': 'Nuclear reactions: specific reactions',
    '27': 'Properties of specific nuclei listed by mass ranges',
    '29': 'Experimental methods and instrumentation for elementary particle and nuclear physics',
    '32': 'Atomic properties and interactions with photons',
    '34': 'Atomic and molecular collision processes and interactions',
    '41': 'Electromagnetism; electron and ion optics',
    '42': 'Optics',
    '47': 'Fluid dynamics',
    '52': 'Physics of plasmas and electric discharges',
    '61': 'Structure of solids and liquids; crystallography',
    '64': 'Equations of state, phase equilibria, and phase transitions',
    '68': 'Surfaces and interfaces; thin films and nanosystems (structure and nonelectronic properties)',
    '71': 'Electronic structure of bulk materials',
    '73': 'Electronic structure and electrical properties of surfaces, interfaces, thin films, and low-dimensional structures',
    '74': 'Superconductivity',
    '75': 'Magnetic properties and materials',
    '78': 'Optical properties, condensed - matter spectroscopy and other interactions of radiation and particles with condensed matter',
    '81': 'Materials science',
    '82': 'Physical chemistry and chemical physics',
    '84': 'Electronics; radiowave and microwave technology; direct energy conversion and storage',
    '87': 'Biological and medical physics',
    '94': 'Physics of the ionosphere and magnetosphere',
    '95': 'Fundamental astronomy and astrophysics; instrumentation, techniques, and astronomical observations',
    '97': 'Stars',
    '98': 'Stellar systems; interstellar medium; galactic and extragalactic objects and systems; the Unive'}
    # authors_to_str()

    # files = 'colabs/wbasic/*0.5*selected_wb_with_author_pac_list.xnet'
    files = 'data/pacs/2lvls/*delta4_v3.xnet'
    #
    files = glob.glob(files)
    files = sorted(files)
    #
    # pac_nets_comm(files)


    # PACS distribuição para 2010
    file = 'data/pacs/2lvls/pac_net_2006_2lvls_delta4_v3_multilevel2.xnet'
    pacs_net_2006 = xnet.xnet2igraph(file)
    data = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')
    papers_2006 = data.vs.select(year_ge=2006, year_le=2010)

    attr_pacs = get_attr_pacs()
    pac_list = get_pac_list()
    comms = dict()
    for pac in pacs_net_2006.vs:
        comms[pac['name']] = pac['community']


    pac_disc = defaultdict(lambda: [])
    paper_count = defaultdict(lambda: set())
    author_count = defaultdict(lambda: set())
    for paper in papers_2006:
        for attr in attr_pacs:
            pac = paper[attr][:2]
            if pac not in pac_list:
                continue
            c = comms[pac]
            paper_count[c].add(paper.index)
            pac_disc[c].append(pac)
            for author in paper['authors_idxs'].split(','):
                author_count[c].add(author)

    print('pac count')

    for comm in pac_disc.keys():
        print(comm)
        dist_pacs = pac_disc[comm]
        u, c = np.unique(dist_pacs, return_counts=True)
        print(sum(c))
        idx = np.argsort(-c)
        pacs = u[idx][:4]
        print(pacs)
        occurs = c[idx][:4]
        authors = author_count[comm]
        plt.bar(np.arange(len(occurs)), occurs, width=0.4)
        plt.xlim((-1, 4))
        plt.xticks(np.arange(len(pacs)), pacs)
        plt.title('pacs distribution; community=%s; \ntotal of authors=%d; total of papers=%d' % (comm, len(authors), len(paper_count[comm])))
        text = ''
        for p in pacs:
            text += '%s: %s' % (p, "\n".join(wrap(code2name[p], 50))) + '\n'
        plt.text(1, 0.7*max(occurs), text[:-1], size=9,
                 ha="left",
                 bbox=dict(boxstyle="round",
                           ec=(0.5, 0.5, 0.5),
                           fc=(0.5, 0.8, 0.8),
                           alpha=0.5
                           )

                 )
        plt.tight_layout()
        plt.savefig('pacs_comm_%s_dist_2010.pdf' % comm)

        plt.clf()


