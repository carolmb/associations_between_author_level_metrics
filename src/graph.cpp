#include "graph.h"

void Graph::print_field(string field) {
	vector<string> *values = get_field_values_str(field);
	if (values) {
		for (vector<string>::iterator it = values->begin() ; it != values->end(); it++) {
			cout << *it << endl;
		}
	} else {
		vector<double> *values = get_field_values_num(field);
		for (vector<double>::iterator it = values->begin() ; it != values->end(); it++) {
			cout << *it << endl;
		}
	}
}

vector<string>* Graph::get_field_values_str(string field) {
	int n_fields = field_names_str.size();
	for (int i = 0; i < n_fields; i++) {
		if (!field_names_str[i].compare(field)) {
			vector<string> *values = &field_values_str[i];
			return values;
		}
	}
	return nullptr;
}

vector<double>* Graph::get_field_values_num(string field) {
	int n_fields = field_names_num.size();
	for (int i = 0; i < n_fields; i++) {
		if (!field_names_num[i].compare(field)) {
			vector<double> *values = &field_values_num[i];
			return values;
		}
	}
	return nullptr;
}

string Graph::tostring() {
	return "Number of vertices " + to_string(n_vtxs) + " Number of edges " + to_string(n_edges) + '\n';
}