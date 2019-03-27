#include "parser.h"

int read_vertices(ifstream &input, vector<string> &names) {
	// header (vertices)
	string description;
	int n_vtxs;
	input >> description;
	input >> n_vtxs;
	input >> description;
	
	// if vertices are named
	string idx;
	input >> quoted(idx);

	if (idx[0] != '#') {
		names.push_back(idx);
		for (int i = 1; i < n_vtxs; i++) {
			input >> quoted(idx);
			names.push_back(idx);
		}
	}
	return n_vtxs;
}

int read_edges(ifstream &input, vector<Edge> &edges, bool &isdirected) {
	string description;
	bool isweighted;
	int n_edges = 0;
	
	input >> description;
	if (!description.compare("#edges")) {
		input >> description;	
	}
	isweighted = !description.compare("weighted");
	input >> description;
	isdirected = !description.compare("directed");
	
	int v0,v1;
	float w;

	input.ignore(); 
	int next = input.peek();
	while (isdigit(next)) {
		input >> v0 >> v1;
		if (isweighted) {
			input >> w;
		}
		Edge e(v0,v1,isdirected,isweighted,w);
		edges.push_back(e);
		n_edges ++;
		
		input.ignore();
		next = input.peek();
	}
	return n_edges;
}

void read_field_values(ifstream &input, int n_elements, string type, vector<double> &field_values) {
	double value;
	
	for (int i = 0; i < n_elements; i++) {
		input >> value;
		field_values.push_back(value);
	}
}

void read_field_values(ifstream &input, int n_elements, string type, vector<string> &field_values) {
	string str;

	for (int i = 0; i < n_elements; i++) {
			
		getline(input,str);
			
		stringstream ss;
		ss << str; 
		ss >> quoted(str);
		
		field_values.push_back(str);
	}
}

void read_extra_fields(ifstream &input, int n_vtxs, int n_edges, 
	vector<string> &field_names_str, vector<vector<string> > &field_values_str, 
	vector<string> &field_names_num, vector<vector<double> > &field_values_num) {
	string header;
	while (input >> header) {
		string field_name;
		input >> quoted(field_name);
		cout << field_name << " ";

		string field_type;
		input >> field_type;
		cout << field_type << endl;

		int n_elements = !header.compare("#v") ? n_vtxs : n_edges;
		
		input.ignore();

		if (!field_type.compare("s")) {
			vector<string> values;
			read_field_values(input,n_elements,field_type,values);
			field_names_str.push_back(field_name);
			field_values_str.push_back(values);
		} else {
			vector<double> values;
			read_field_values(input,n_elements,field_type,values);
			field_names_num.push_back(field_name);
			field_values_num.push_back(values);
		}
	}
}

Graph* xnet2graph(string filename) {
    ifstream input (filename);
    if (input.is_open()) {

    	vector<string> names;
    	int n_vtxs = read_vertices(input,names);

    	vector<Edge> edges;
    	bool isdirected = false;
    	int n_edges = read_edges(input,edges,isdirected);

    	vector<string> field_names_str;
    	vector<vector<string> > field_values_str;
    	if (names.size()) {
    		field_names_str.push_back("name");
    		field_values_str.push_back(names);
    	}

    	vector<string> field_names_num;
    	vector<vector<double> > field_values_num;

    	read_extra_fields(input,n_vtxs,n_edges,
    		field_names_str,field_values_str,field_names_num,field_values_num);

		Graph *g = new Graph(n_vtxs,n_edges,isdirected,edges,
			field_names_str,field_values_str,field_names_num,field_values_num);

        input.close();

        return g;
    } else {
        cout << "Error during file reading." << endl;
    }
}


