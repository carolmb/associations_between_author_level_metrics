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

vector<vector<string> > split_vec(vector<string> *vec, char d) {
	vector<vector<string> > output;
	for (vector<string>::iterator it = vec->begin(); it != vec->end(); it++) {
		vector<string> current = split_str(*it, d);
		output.push_back(current);
	}
	return output;
}

template<class T>
set<T> get_unique_values(vector<vector<T> > &values) {
	
	set<T> unique;
	auto set_it = unique.begin(); 

	typename vector<vector<T> >::iterator it = values.begin();
	
	while(it != values.end()) {
	    typename vector<T>::iterator val_it = it->begin();
	    
	    while(val_it != it->end()) {
	    	
			set_it = unique.insert(set_it,*val_it);
			val_it ++;
		}
		it ++;

	}
	cout << "Unique authors size " << unique.size() << endl;
	return unique;
}

igraph_t* create_colab(igraph_t *g, int year_begin, int year_end) {
 //    g->remove_vtx_by("year","gt",year_begin);
	// vector<string> *authors = g->get_field_values("author_idxs");
	// vector<string> *paper_idxs = g->get_field_values("id");
	
 //    vector<vector<string> > authors_vec = split_vec(authors,';');
 //    set<string> unique_authors = get_unique_values(authors_vec);

 //    long int total_authors = unique_authors.size();

    return nullptr;
}