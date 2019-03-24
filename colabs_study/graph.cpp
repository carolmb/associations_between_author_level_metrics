#include "graph.h"

void Graph::print_field(string field) {
	vector<Type> *values = get_field_values(field);
	for (vector<Type>::iterator it = values->begin() ; it != values->end(); it++) {
		cout << get_Type_string(*it) << endl;
	}
}

vector<Type>* Graph::get_field_values(string field) {
	int n_fields = field_names.size();
	for (int i = 0; i < n_fields; i++) {
		if (!field_names[i].compare(field)) {
			vector<Type> *values = &field_values[i];
			return values;
		}
	}
}