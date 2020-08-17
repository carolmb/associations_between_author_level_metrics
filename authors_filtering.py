import xnet
import json
import glob
import numpy as np

from igraph import *
from collections import defaultdict
from util import save, load

from util import get_attr_pacs, get_pac_list, get_pacs_paper_published

attr_pacs = get_attr_pacs()
pac_list = get_pac_list()


def get_in(paper, data):
    pacs = []
    p_neighbors = data.neighbors(paper, mode=IN)
    n_neighbors = len(p_neighbors)
    return n_neighbors


def get_out(paper, data):
    pacs = []
    p_neighbors = data.neighbors(paper, mode=OUT)
    n_neighbors = len(p_neighbors)
    return n_neighbors


def get_and_save_freq_of(data, get_freq, delta, filename):
    history = defaultdict(lambda: defaultdict(lambda: 0))

    year_begin = 1986
    year_end = 2006

    for i, year in enumerate(range(year_begin, year_end + 1)):
        print("current year %d" % year)
        subset = data.vs.select(year_ge=year, year_le=year + delta)
        subgraph = data.subgraph(subset)
        for paper in subgraph.vs:
            authors_idxs = paper['authors_idxs'].split(',')
            value = get_freq(paper, subgraph)
            for author in authors_idxs:
                history[author][year + delta] += value
        save(history, filename)

    return history


def get_and_save_freq_of_by_paper(data, get_freq, delta, filename):
    history = defaultdict(lambda: defaultdict(lambda: 0))

    year_begin = 1986
    year_end = 2006

    for i, year in enumerate(range(year_begin, year_end + 1)):
        author_freq = defaultdict(lambda: 0)
        print("current year %d" % year)
        subset = data.vs.select(year_ge=year, year_le=year + delta)
        subgraph = data.subgraph(subset)
        for paper in subgraph.vs:
            authors_idxs = paper['authors_idxs'].split(',')
            value = get_freq(paper, subgraph)
            for author in authors_idxs:
                history[author][year + delta] += value
                author_freq[author] += 1

        keys = history.keys()
        for a in keys:
            if author_freq[a] != 0:
                history[a][year + delta] = history[a][year + delta]/author_freq[a]
        save(history, filename)

    return history

'''
    essa função diz a quantidade de autores por ano
    e faz rastreio para que seja os mesmos autores
'''


def get_min_freqs_author_track(data, min_values):
    authors_by_min = dict()
    for min_value in min_values:
        freq = defaultdict(lambda: [])
        freq_all_authors = defaultdict(lambda: 0)
        last_year = set(data.keys())
        for author, a_hist in data.items():
            for year, value in a_hist.items():
                freq_all_authors[year] += 1
                if int(year) < 1995:  # ignora tudo que acontece antes de 2000
                    continue
                if value < min_value:
                    last_year.discard(author)
                else:
                    freq[author].append(int(year))
        #         x = list(freq_all_authors.keys())
        #         y = list(freq_all_authors.values())
        #         plt.plot(x,y)

        valids = []
        for author, years in freq.items():
            if author not in last_year:  # last_year começa com todos os autores e remove os com menos do que o min
                continue
            min_year = min(years)
            max_year = max(years)
            seq_years = set(range(min_year, max_year + 1))
            inter = set(seq_years).intersection(years)  # garante a sequencia sem furos
            if len(inter) == len(seq_years) and 1995 in seq_years and 2010 in seq_years:  # entre 2000 e 2010
                valids.append(author)
        authors_by_min[min_value] = valids
        print(len(valids))
    return authors_by_min


'''
    seleção de autores por critério frouxo
    min 10 pontos que não sejam 0 citações ou referências
'''


def get_select_author_min_criteria(data, min_non_zero=10, min_v=1):
    valid = set()

    for author, a_hist in data.items():
        number_of_non_zero = 0
        for year, value in a_hist.items():
            if int(year) < 1995:  # ignora tudo que acontece antes de 2000
                continue
            if value >= min_v:
                number_of_non_zero += 1

        if number_of_non_zero >= min_non_zero:  # and '1995' in a_hist and '2010' in a_hist:
            valid.add(author)

    return valid


'''
    calcula o total de citações ou referências
'''


def get_total_of(data, get_out, filename):
    freqs = defaultdict(lambda: 0)
    for paper in data.vs:
        authors_idxs = paper['authors_idxs'].split(',')
        value = get_out(paper, data)
        for author in authors_idxs:
            freqs[author] += value
    save(freqs, filename)
    return freqs


'''
    NAO É MAIS UTILIZADO
    essa função diz a quantidade de autores por ano,
    sem fazer rastreio para que seja os mesmos autores
'''


def get_min_freqs(data, min_values):
    freq = defaultdict(lambda: [])
    for author, a_hist in data.items():
        for year, value in a_hist.items():
            freq[year].append(value)

    years = []
    freqs_dict = {}
    for min_value in min_values:
        freqs_dict[min_value] = []

    for year, freqs in freq.items():
        years.append(year)
        freqs = np.asarray(freqs)
        total = len(freqs)

        for min_value in min_values:
            p = sum(freqs >= min_value)
            freqs_dict[min_value].append(p)

        return years, freqs_dict


'''
    seleciona o top x pelo critério mínimo
    ou pelo total de items que deve ter o ranking

    exemplo de uso:
    years = ['1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010']
    ranking_in_by_year = get_ranking_by_year(authors_in,years,25,500)
    ranking_out_by_year = get_ranking_by_year(authors_out,years,25)
    save(ranking_in_by_year,'data2/authors_in_ranking500_by_year.json')

    ranking_by_year = dict()
    for year,ranking_in in ranking_in_by_year.items():
        ranking_in = set(ranking_in)
        ranking_out = set(ranking_out_by_year[year])
        ranking = ranking_in & ranking_out
        ranking_by_year[year] = list(ranking)
    # quando eu queria calcular o top 25 citações e referências, MAS SEM RASTREIO 
'''


def get_ranking_by_year(authors_in, years, min_value, k):
    ranking_by_year = dict()
    for year in years:
        ranking = []
        for a, a_hist in authors_in.items():
            if year in a_hist:
                count = a_hist[year]
                ranking.append((count, a))

        if k > 0:
            ranking = sorted(ranking, reverse=True)
            ranking = ranking[:k]
            authors = [a for count, a in ranking]
        else:
            authors = []
            for c, a in ranking:
                if c >= min_value:
                    authors.append(a)

        print(len(authors) * 100 / len(ranking))
        ranking_by_year[year] = authors
    return ranking_by_year


if __name__ == '__main__':

    output_folder = 'data2\\'

    data = xnet.xnet2igraph('data/citation_network_ge1985_pacs.xnet')

    filenames = sorted(glob.glob('data/pacs/2lvls/*_multilevel2.xnet'))
    pac_nets = []
    for filename in filenames:
        net = xnet.xnet2igraph(filename)
        pac_nets.append(net)

    # citações e referências por intervalo de tempo PASSADO

    get_and_save_freq_of_by_paper(data, get_in, 4, output_folder + 'authors_in_freq_by_paper.json')
    get_and_save_freq_of_by_paper(data, get_out, 4, output_folder + 'authors_out_freq_by_paper.json')

    authors_out = load('data2\\authors_out_freq.json')
    authors_in = load('data2\\authors_in_freq.json')

    # seleção dos autores conforme os critérios mais exigentes e menos exigentes

    min_values = [10, 25, 50, 75, 100]
    authors_valid_out = get_min_freqs_author_track(authors_out, min_values)
    authors_valid_in = get_min_freqs_author_track(authors_in, min_values)

    for min_value in min_values:
        inter = set(authors_valid_in[min_value]) & set(authors_valid_out[min_value])
        print(min_value, len(inter))

    for min_v, non_zero in [(10, 10), (5, 10), (5, 16), (1, 10), (10, 16), (25, 16), (50, 16)]:
        # for min_v,non_zero in ]:
        authors_in_min_criteria = get_select_author_min_criteria(authors_in, non_zero, min_v)
        authors_out_min_criteria = get_select_author_min_criteria(authors_out, non_zero, min_v)
        valid_authors_min_in_out = authors_out_min_criteria & authors_in_min_criteria
        print(len(valid_authors_min_in_out))
        file_to_save = open(output_folder + 'valid_authors_min_criteria_in_out_%d_%d_temp.txt' % (min_v, non_zero), 'w')
        for author in valid_authors_min_in_out:
            file_to_save.write(author + '\n')
        file_to_save.close()

    # numero de referências e citações por ano

    # get_total_of(data,get_out,output_folder+num_of_refs.json')
    # get_total_of(data,get_in,output_folder+'num_of_citations.json')
