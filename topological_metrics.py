import numpy as np
from igraph import *
import xnet
import glob
# import matplotlib.pyplot as plt
import json
import math
import concurrent.futures

np.set_printoptions(precision=6)

def metric_infos_to_json(name,metric_values,json_dict):
    values = np.asarray(metric_values)
    metric_dict = dict()
    metric_dict['name'] = name
    metric_dict['min'] = str(min(values))
    metric_dict['max'] = str(max(values))
    metric_dict['mean'] = str(np.mean(values))
    metric_dict['std'] = str(np.std(values))
    json_dict['metrics'].append(metric_dict)

    return json_dict

def graph_infos_to_json(year,graph,original_graph,weight_name):
    json_dict = dict()
    json_dict['year'] = year
    json_dict['metrics'] = []

    norm = np.asarray(original_graph.es[weight_name])
    max_v = max(norm)
    min_v = min(norm)
    norm = (norm - min_v)/(max_v - min_v)
    dist = [math.sqrt(2*(1.01-w)) for w in norm]

    names = set(graph.vs['name'])
    vtxs = original_graph.vs.select(name_in=names)

    degree = original_graph.degree(vtxs)
    json_dict = metric_infos_to_json('degree',degree,json_dict)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_v = {executor.submit(original_graph.betweenness, v, weights=dist, cutoff=20) : v for v in vtxs}
        for future in concurrent.futures.as_completed(future_to_v):
            v = future_to_v[future]
            try:
                b = future.result()
                v['betweenness'] = b
            except Exception as exc:
                print('%r generated an exception: %s' % (v, exc))
        
    json_dict = metric_infos_to_json('betweenness',vtxs['betweenness'],json_dict)

    #closeness = original_graph.closeness(vtxs,weights=dist,cutoff=20)
    #json_dict = metric_infos_to_json('closeness',closeness,json_dict)

    return json_dict


headers = ['colabs/basic_colab_cut/*deleted_basic.xnet','colabs/basic_colab_cut/*selected_basic.xnet']

filenames_seq = []

for header in headers:
	filenames = glob.glob(header)
	filenames = sorted(filenames)
	filenames_seq.append(filenames)

for i,filenames in enumerate(filenames_seq):
    current_year = 1990
    all_years_metrics = dict()
    all_years_metrics['infos'] = []

    for filename in filenames:

        year = current_year
        print(year,'(',filename,')')
        current_year += 1

        graph = xnet.xnet2igraph(filename)
        original_graph = xnet.xnet2igraph('colabs/original/colab_'+str(year)+'_'+str(year+4)+'_test')
        json_dict = graph_infos_to_json(year,graph,original_graph,'weight_basic')
        all_years_metrics['infos'].append(json_dict)

    json_obj = json.dumps(all_years_metrics)

    colab_type = headers[i].split('/')[1]
    edge_type = headers[i].split('/')[2][1:-5]
    base = colab_type + '_' + edge_type

    output = open("graphs_metrics_"+base+".json",'w')
    output.write(json_obj)
    output.close()

# filename = 'colab_1990_1994_test_0.8_selected_basic.xnet'
# graph = xnet.xnet2igraph(filename)
# graph_infos_to_json(1990,graph,graph,'weight_basic')