import util
import xnet, glob
import pandas as pd
import igraph as ig
import multiprocessing
from collections import defaultdict
from util import get_pac_comm
from util import get_pacs_in, get_pacs_out
import temporal_series

def authors_dois():
    """
    Gera arquivo com a lista de DOIS de cada autor por intervalo de tempo
    janela 5 anos, de 1991 a 2006 (terminando em 2010)
    Returns
    -------

    """
    net = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')
    sinatra_aps = pd.read_csv('sinatra/APS_author2DOI.dat', sep=',')

    doi_year = defaultdict(lambda: set())
    for doi, year in zip(net.vs['id'], net.vs['year']):
        doi_year[year].add(doi.lower())

    doi_year_delta = defaultdict(lambda: set())
    delta = 4
    for year in range(1991, 2007):
        for i in range(5):
            doi_year_delta[year] |= doi_year[year + i]

    author_papers = defaultdict(lambda: 0)
    valid_authors = []
    for idx, row in sinatra_aps.iterrows():
        authors_publications = set(row[2].lower().split("\t"))
        author_history = []

        count = 0
        for year, papers in doi_year_delta.items():
            intersect = papers & authors_publications
            author_history.append(intersect)
            author_papers[row[0]] += len(doi_year[year] & authors_publications)

        valid_authors.append([row[0], author_history])

    output_file = open("sinatra/authors_publications_per_window.txt", 'w')
    for row in valid_authors:
        output_file.write(str(row[0]) + '\t')
        for year_dois in row[1]:
            output_file.write(','.join(list(year_dois)) + '\t')
        output_file.write('\n')
    output_file.close()

    util.save(author_papers, 'sinatra/authors_total_of_papers_1991_2010.json')


def cits_past_past():
    citation_net = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')
    paper_infos = defaultdict(lambda: dict())

    delta = 4
    for year in range(1991, 2007):
        subset = citation_net.vs.select(year_ge=year, year_le=year + delta)
        subgraph = citation_net.subgraph(subset)
        for paper in subgraph.vs:
            doi = paper['id'].lower()
            paper_infos[year + delta][doi] = dict()
            paper_infos[year + delta][doi]['cits'] = len(subgraph.neighbors(paper, mode=ig.IN))
            paper_infos[year + delta][doi]['refs'] = len(subgraph.neighbors(paper, mode=ig.OUT))

    return paper_infos


def get_citation_future(paper, future_interval, MODE):
    citations = []
    for neighbor in paper.neighbors(mode=MODE):
        if future_interval[0] <= neighbor['year'] <= future_interval[1]:
            citations.append(neighbor)
    return citations


def cits_past_fut(delta_future=3):
    citation_net = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')
    paper_infos = defaultdict(lambda: dict())

    delta = 4
    for year in range(1991, 2007):
        subset = citation_net.vs.select(year_ge=year, year_le=year + delta)
        future_interval = (year + delta + 1, year + delta + 1 + delta_future)

        for paper in subset:
            doi = paper['id'].lower()
            paper_infos[year + delta][doi] = dict()
            paper_infos[year + delta][doi]['cits'] = len(get_citation_future(paper, future_interval, ig.IN))

    return paper_infos


def get_authors_cits_refs(authors_dois):
    paper_infos = cits_past_past()
    paper_infos_fut = cits_past_fut()

    authors_history = dict()
    for idx, row in authors_dois.iterrows():
        author_history = dict()
        year = 1995
        for _, dois_year in row[1:-1].iteritems():
            author_history[year] = {'cits': 0, 'refs': 0, 'cits_fut': 0, 'n_papers': 0}
            if type(dois_year) == str:
                dois_year = dois_year.split(',')
                for doi in dois_year:
                    c = paper_infos[year][doi]['cits']
                    author_history[year]['cits'] += c
                    r = paper_infos[year][doi]['refs']
                    author_history[year]['refs'] += r
                    cf = paper_infos_fut[year][doi]['cits']
                    author_history[year]['cits_fut'] += cf
                    author_history[year]['n_papers'] += 1
            year += 1

        c10 = 0
        r10 = 0
        for year, infos in author_history.items():
            if infos['cits'] >= 10:
                c10 += 1
            if infos['refs'] >= 10:
                r10 += 1

        if c10 >= 10 and r10 >= 10:
            authors_history[row[0]] = author_history

    return authors_history


def pacs_per_paper():
    filenames = sorted(glob.glob('data/pacs/2lvls/*_delta4_v3_multilevel2.xnet'))
    pac_nets = dict()
    delta = 4
    year = 1991
    for filename in filenames:
        net = xnet.xnet2igraph(filename)
        pac_nets[year + delta] = net
        year += 1

    history_pacs_in = defaultdict(lambda: defaultdict(lambda: {}))
    history_pacs_out = defaultdict(lambda: defaultdict(lambda: {}))

    data = xnet.xnet2igraph('data/citation_network_ge1991_pacs.xnet')

    year_begin = 1991
    year_end = 2006
    delta = 4

    for year in range(year_begin, year_end + 1):
        print("current year %d" % year)
        subset = data.vs.select(year_ge=year, year_le=year + delta)
        subgraph = data.subgraph(subset)
        for paper in subgraph.vs:
            comms_by_paper = get_pac_comm(pac_nets[year + delta], paper, subgraph, get_pacs_in)
            history_pacs_in[year + delta][paper['id'].lower()] = comms_by_paper

            comms_by_paper = get_pac_comm(pac_nets[year + delta], paper, subgraph, get_pacs_out)
            history_pacs_out[year + delta][paper['id'].lower()] = comms_by_paper

    return history_pacs_in, history_pacs_out


def authors_temporal_series(authors_dois, authors_cit_ref, papers_pacs_in, papers_pacs_out):
    valid_authors = set(list(authors_cit_ref.keys()))
    print(len(valid_authors))
    author_pacs_history_in = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    author_pacs_history_out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    for _, row in authors_dois.iterrows():
        author = row[0]
        if author not in valid_authors:
            continue
        year = 1995
        for _, dois in row[1:-1].iteritems():
            if type(dois) == str:
                _dois = dois.split(',')
                for doi in _dois:
                    c_in = papers_pacs_in[year][doi]
                    c_out = papers_pacs_out[year][doi]

                    for comms, n_comms in c_in:
                        for comm in comms:
                            author_pacs_history_in[author][year][comm] += 1 / n_comms

                    for comms, n_comms in c_out:
                        for comm in comms:
                            author_pacs_history_out[author][year][comm] += 1 / n_comms
            year += 1

    for year in range(1995, 2010):
        util.norm(author_pacs_history_in, year)
        util.norm(author_pacs_history_out, year)

    util.save(util.authors_metric(author_pacs_history_in, util.get_div), 'sinatra/div_in_past.json')
    util.save(util.authors_metric(author_pacs_history_out, util.get_div), 'sinatra/div_out_past.json')


def save_jsons(authors_history):

    cits = dict()
    refs = dict()
    cits_fut = dict()
    papers = dict()
    for author, hist in authors_history.items():
        cits[author] = dict()
        refs[author] = dict()
        cits_fut[author] = dict()
        papers[author] = dict()
        for year, values in hist.items():
            papers[author][year] = values['n_papers']
            if papers[author][year] == 0:
                papers[author][year] = 0.2
            cits[author][year] = values['cits']/papers[author][year]
            refs[author][year] = values['refs']
            cits_fut[author][year] = values['cits_fut']/papers[author][year]

    util.save(cits, 'sinatra/cits_pastpast.json')
    util.save(refs, 'sinatra/refs_pastpast.json')
    util.save(cits_fut, 'sinatra/cits_futpast.json')
    util.save(papers, 'sinatra/papers_past.json')


if __name__ == '__main__':
    # authors_dois()
    authors_dois = pd.read_csv('sinatra/authors_publications_per_window.txt', sep='\t', header=None)
    authors_history = get_authors_cits_refs(authors_dois)  # valid authors only

    save_jsons(authors_history)
    history_pacs_in, history_pacs_out = pacs_per_paper()
    authors_temporal_series(authors_dois, authors_history, history_pacs_in, history_pacs_out)

    authors_out = util.load('sinatra/refs_pastpast.json')
    authors_in = util.load('sinatra/cits_pastpast.json')  # citações
    authors_number_of_papers = util.load('sinatra/papers_past.json')

    # ESSES SÃO OS REGISTROS DO LADO DIREITO
    cit_from_citations = util.load('sinatra/cits_futpast.json')  # TODO aqui é _per_paper

    authors_in_div = util.load('sinatra/div_in_past.json')
    authors_out_div = util.load('sinatra/div_out_past.json')

    valid_authors = set(list(authors_history.keys()))
    authors_total_papers = util.load('sinatra/authors_total_of_papers_1991_2010.json')
    paper_count_1991 = {author: npapers for author, npapers in authors_total_papers.items() if int(author) in valid_authors}
    authors_by_paper_count_1991 = temporal_series.get_authors_by_percentile(paper_count_1991, 'paper_')

    files_valid_authors = {**authors_by_paper_count_1991}

    files = list(files_valid_authors.keys())
    print(files)
    temporal_series_div_out = dict()
    temporal_series_div_in = dict()
    # temporal_series_in = dict()
    temporal_series_in_abs = dict()
    # temporal_series_out = dict()
    temporal_series_numb_papers = dict()

    # future_from_to_cit_abs = dict()
    future_from_to_cit = dict()
    # future_from_to_div = dict()
    # future_out_to_to_ref = dict()
    # future_out_to_to_div = dict()
    # future_number_of_papers = dict()

    for file, valid_authors in files_valid_authors.items():
        temporal_series_div_out[file] = temporal_series.get_temporal_series(valid_authors, authors_out_div)
        temporal_series_div_in[file] = temporal_series.get_temporal_series(valid_authors, authors_in_div)
        # temporal_series_in[file] = temporal_series.get_temporal_series(valid_authors, authors_in_per_paper)
        temporal_series_in_abs[file] = temporal_series.get_temporal_series(valid_authors, authors_in)
        # temporal_series_out[file] = temporal_series.get_temporal_series(valid_authors, authors_out_per_paper)
        temporal_series_numb_papers[file] = temporal_series.get_temporal_series(valid_authors, authors_number_of_papers)

        # future_from_to_cit_abs[file] = temporal_series.get_temporal_series(valid_authors, cit_from_abs_citations)
        future_from_to_cit[file] = temporal_series.get_temporal_series(valid_authors, cit_from_citations)
        # future_from_to_div[file] = temporal_series.get_temporal_series(valid_authors, cit_from_diversity)
        # future_out_to_to_ref[file] = temporal_series.get_temporal_series(valid_authors, out_to_to_citations)
        # future_out_to_to_div[file] = temporal_series.get_temporal_series(valid_authors, out_to_to_diversity)
        # future_number_of_papers[file] = temporal_series.get_temporal_series(valid_authors, authors_number_of_papers_fut)

    pool = multiprocessing.Pool(processes=8)

    # random_state = RandomState(seed=9)
    for file in files:
        print(file)

        # temporal_series.corr_temporal_series(pool, *temporal_series.filter_const(temporal_series_div_out[file], future_from_to_cit[file]),
        #                      'Correlation between div out(PAST->PAST) and cit(FUT->PAST)',
        #                      'corr_div_out_past_cit_from_fut_%s_1991' % file)

        # temporal_series.corr_temporal_series(pool, *temporal_series.filter_const(temporal_series_div_out[file], future_from_to_div[file]),
        #                      'Correlation between div out(PAST->PAST) and div in(FUT->PAST)',
        #                      'corr_div_out_past_div_in_fut_%s_1991' % file)

        # temporal_series.corr_temporal_series(pool, *temporal_series.filter_const(temporal_series_div_in[file], future_from_to_cit[file]),
        #                      'Correlation between div in(PAST->PAST) and cit(FUT->PAST)',
        #                      'corr_div_in_past_cit_from_fut_%s_1991' % file)

        # temporal_series.corr_temporal_series(pool, *filter_const(temporal_series_in[file], future_from_to_div[file]),
        #                      'Correlation between cit(PAST->PAST) and div in(FUT->PAST)',
        #                      'corr_cit_past_div_in_fut_%s_1991' % file)

        temporal_series.corr_temporal_series(pool, *temporal_series.filter_const(temporal_series_numb_papers[file], future_from_to_cit[file]),
                             'Correlation between papers(PAST->PAST) and cit(FUT->PAST)',
                             'corr_papers_past_cit_from_fut_%s_1991' % file)

        # temporal_series.corr_temporal_series(pool, *temporal_series.filter_const(temporal_series_numb_papers[file], future_from_to_div[file]),
        #                      'Correlation between papers(PAST->PAST) and div in(FUT->PAST)',
        #                      'corr_papers_past_div_in_fut_%s_1991' % file)