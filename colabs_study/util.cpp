#include "util.h"

bool is_alpnum(string str) {
	return find_if(str.begin(),str.end(),[](char c) { return !(isalnum(c) || (c == ' ') || (c == '.')); }) == str.end();
}

vector<string> split_str(string &str, char d) {
	istringstream ss(str);
	string token;

	vector<string> output;
	while(getline(ss, token, d)) {
		output.push_back(token);
	}
	return output;
}

vector<vector<string> > string2vectorofvector(vector<string> *vec, char d) {
	vector<vector<string> > output;
	for (vector<string>::iterator it = vec->begin(); it != vec->end(); it++) {
		vector<string> current = split_str(*it, d);
		output.push_back(current);
	}
	return output;
}

void get_unique_authors(vector<vector<string> > &authors_name_values,
	vector<vector<string> > &authors_idx_values,
	set<pair<string,long long int> > &unique_authors) {
	
	auto set_it = unique_authors.insert(unique_authors.begin(), make_pair("-1",-1)); 

	vector<vector<string> >::iterator name_vec_it = authors_name_values.begin();
	vector<vector<string> >::iterator idx_vec_it = authors_idx_values.begin();
	
	while(name_vec_it != authors_name_values.end() && idx_vec_it != authors_idx_values.end()) {
	    vector<string>::iterator name_it = name_vec_it->begin();
	    vector<string>::iterator idx_it = idx_vec_it->begin();
	    
	    while(name_it != name_vec_it->end() && idx_it != idx_vec_it->end()) {
			pair<string, long long int> p = make_pair(*name_it,stod(*idx_it));
			set_it = unique_authors.insert(set_it,p);
			name_it ++;
			idx_it ++;
		}
		name_vec_it ++;
		idx_vec_it ++;

	}
	cout << "Unique authors size " << unique_authors.size() << endl;
}

void desambiguation(Graph *g) {
	vector<string> *authors_name = g->get_field_values_str("authors_name");
    vector<string> *authors_idx = g->get_field_values_str("authors_idx");
    vector<vector<string> > authors_name_values = string2vectorofvector(authors_name, ';');
    vector<vector<string> > authors_idx_values = string2vectorofvector(authors_idx, ',');

	set<pair<string,long long int> > unique_authors;
    get_unique_authors(authors_name_values, authors_idx_values, unique_authors);

    int i = 0;
    map<string,set<string> > roots;
    for (set<pair<string,long long int> >::iterator it = unique_authors.begin(); it != unique_authors.end(); it++) {
    	string name = it->first;
    	if (!is_alpnum(name)) {
    		i ++;
    		continue;
    	}
    	vector<string> name_spl = split_str(name, ' ');
		if (name_spl.size() > 1) {
			string key = name_spl[name_spl.size()-1] + ". " + name_spl[0][0] + ".";
			
			string middle = "";
			for (int i = 1; i < name_spl.size()-1; i++) {
				middle += name_spl[i][0] + ". ";
			}
			if (roots.find(key) == roots.end()) {
				roots[key] = set<string>();
			}
			if (middle.size()) {
				roots[key].insert(middle);
			}
		} else {
			i ++;
		}
    }

    for (map<string,set<string> >::iterator it = roots.begin(); it != roots.end(); it++) {
    	cout << it->first << ": ";
    	for (set<string>::iterator mid = (it->second).begin(); mid != (it->second).end(); mid++) {
    		cout << *mid << ", ";
    	}
    	cout << (it->second).size() << endl;
    }

    cout << "Total roots " << roots.size() << endl; 
    cout << "Total names with 1 word len " << i << endl;
}