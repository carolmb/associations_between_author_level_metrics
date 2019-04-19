#include "util.h"

bool is_alpnum(string str) {
	return find_if(str.begin(),str.end(),[](char c) { return !(isalnum(c) || (c == ' ') || (c == '.')); }) == str.end();
}

vector<string> split_str(string &str,  char d) {
	istringstream ss(str);
	string token;

	vector<string> output;
	while(getline(ss, token, d)) {
		output.push_back(token);
	}

	return output;
}

vector<vector<long long int> > split_vec(vector<string> *vec, char d) {
	vector<vector<long long int> > output;
	for (vector<string>::iterator it = vec->begin(); it != vec->end(); it++) {
		vector<string> current = split_str(*it, d);
		vector<long long int> current_conv;
		for (int i = 0; i < current.size(); i++) {
			long long int p_idx = stoll(current[i],nullptr,10);
			current_conv.push_back(p_idx);
		}
		output.push_back(current_conv);
	}
	return output;
}

template<class T>
set<T> get_unique_values(vector<vector<T> > &values,vector<int> &sizes) {
	
	set<T> unique;
	auto set_it = unique.begin(); 

	typename vector<vector<T> >::iterator it = values.begin();
	
	while(it != values.end()) {
	    int size = it->size();
	    sizes.push_back(size);
	    typename vector<T>::iterator val_it = it->begin();
	    
	    while(val_it != it->end()) {
	    	
			set_it = unique.insert(set_it,*val_it);
			val_it ++;
		}
		it ++;

	}
	return unique;
}

/*
igraph_vector_t select_ids(igraph_t &g, int &begin, int &end) {
	igraph_vector_t years,idxs;
	VANV(&g,"year",&years);
	int size = igraph_vector_size(&years);
	for (int i = 0; i < size; i++) {
		if (years[i] >= begin and years[i] <= end) {
			igraph_vector_push_back(&idxs,i);
		}
	}
	return idxs;
}

igraph_t create_colab(igraph_t &g, int year_begin, int year_end) {

	igraph_t g_copy = igrapy_copy(&g);
	igraph_vector_t idxs_to_del = select_ids(g,year_begin,year_end);
	igraph_delete_vertices(&g,idxs_to_del);
	g->remove_vtx_by("year","gt",year_begin);
	vector<string> *authors = g->get_field_values("author_idxs");
	vector<string> *paper_idxs = g->get_field_values("id");

	vector<vector<string> > authors_vec = split_vec(authors,';');
	set<string> unique_authors = get_unique_values(authors_vec);

	long int total_authors = unique_authors.size();

	return nullptr;
}
*/

bool comp (int i,int j) { return (i<j); }

void print_papers_stat(igraph_t &g,igraph_t &citation_net) {
	time_t t0;
    time(&t0);
    igraph_strvector_t papers;
	igraph_strvector_init(&papers,igraph_ecount(&g));
	EASV(&g,"papers",&papers);

	vector<string> papers_strings;
	int size = igraph_strvector_size(&papers);
	for (int i = 0; i < size; i++) {
		string str1;
		str1 = STR(papers,i);
		str1 = str1.substr(1,str1.size()-2);
		papers_strings.push_back(str1);
	}
	
	vector<vector<long long int> > papers_splited = split_vec(&papers_strings, ',');
	vector<int> sizes;
	set<long long int> unique_papers = get_unique_values(papers_splited, sizes);
	cout << "Total of unique papers " << unique_papers.size() << endl;

	igraph_vector_t times_cited; 
	igraph_vector_init(&times_cited,igraph_vcount(&citation_net));
	VANV(&citation_net,"times_cited",&times_cited);

	vector<int> times_cited_each_paper;
	for(set<long long int>::iterator it = unique_papers.begin(); it != unique_papers.end(); it++) {
		times_cited_each_paper.push_back(VECTOR(times_cited)[*it]);
	}
	sort (times_cited_each_paper.begin(), 
		times_cited_each_paper.end(), 
		comp);
	double sum = 0.0;
	size = times_cited_each_paper.size();
	for (vector<int>::iterator it = times_cited_each_paper.begin(); it != times_cited_each_paper.end(); it++) {
		sum += *it;
	}
	double mean = sum/size;
	double meadian = times_cited_each_paper[(int)size/2];
	cout << "Mean " << mean << " Median " << meadian << endl;
    time_t t1;
    time(&t1);
    cout << "print_paper_stat run time " << difftime(t1,t0) << "s\n";
    // TODO mean and median for sizes
}

igraph_t get_giant_component(igraph_t &g) {
	int current_size = 0;
	int max_size = 0;
	int max_idx = -1;

	igraph_vector_ptr_t components;
    igraph_vector_ptr_init(&components,0);
    igraph_decompose(&g, &components, IGRAPH_WEAK, -1, 2);
    for (int i = 0; i < igraph_vector_ptr_size(&components); i++) {
        igraph_t *component = (igraph_t *) igraph_vector_ptr_e(&components,i);
        current_size = igraph_vcount(component);
        if (current_size > max_size) {
        	max_size = current_size;
        	max_idx = i;
        }
    }
    igraph_t *component = (igraph_t *) igraph_vector_ptr_e(&components,max_idx);
    igraph_t giant_component = *component;
    igraph_decompose_destroy(&components);
    return giant_component;
}