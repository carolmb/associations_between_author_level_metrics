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

int read_edges(ifstream &input, vector<Edge> &edges, vector<double> &weights, bool &isdirected) {
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
			weights.push_back(w);
		}
		Edge e(v0,v1);
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

void read_extra_fields(igraph_t &g, ifstream &input, int n_vtxs, int n_edges) {
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
			
		} else {
			vector<double> values;
			read_field_values(input,n_elements,field_type,values);
			
		}
	}
}

void add_edges(igraph_t &g, int n_edges, vector<Edge> &p_edges, vector<double> &weights) {
	igraph_vector_t edges;
	igraph_vector_init(&edges, n_edges);

	for(vector<Edge>::iterator it = p_edges.begin(); it != p_edges.end(); it++) {
		int s = it->source;
		int t = it->target;
		VECTOR(edges)[s] = t;	
	}
	
	igraph_add_edges(&g,&edges,0);

	if (weights.size()) {
		igraph_vector_t ws;
		igraph_vector_init(&ws, n_edges);
		int pos = 0;
		for(vector<double>::iterator it = weights.begin(); it != weights.end(); it++) {
			VECTOR(ws)[pos]=*it;
			pos++;
		}
		igraph_cattribute_EAN_setv(&g,"weight",&ws);
	}
}

void add_string_attr_vtx(igraph_t &g, vector<string> &attr, string field) {
	if (!attr.size())
		return;
	igraph_strvector_t values;
	igraph_strvector_init(&values,attr.size());
	int pos = 0;
	for(vector<string>::iterator it = attr.begin(); it != attr.end(); it++) {
		igraph_strvector_set(&values,pos,it->c_str());
		pos++;
	}
	igraph_cattribute_VAS_setv(&g,field.c_str(),&values);
}

igraph_t* xnet2igraph(string filename) {
    ifstream input (filename);
    if (input.is_open()) {

    	vector<string> names;
    	int n_vtxs = read_vertices(input,names);

    	vector<Edge> edges;
    	vector<double> weights;
    	bool isdirected = false;
    	int n_edges = read_edges(input,edges,weights,isdirected);

		igraph_t g;		
		igraph_empty(&g,n_vtxs,isdirected);
		add_string_attr_vtx(g,names,"names");
		add_edges(g,n_edges,edges,weights);

		read_extra_fields(g,input,n_vtxs,n_edges);

        input.close();

        return nullptr;
    } else {
        cout << "Error during file reading." << endl;
    }
    return nullptr;
}


