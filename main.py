from graph import Graph


my_graph = Graph([1,2,3,4,5,6,7,8])

my_graph.add_edge_weight(1,2,1)
my_graph.add_edge_weight(1,3,1)
my_graph.add_edge_weight(2,3,1)
my_graph.add_edge_weight(3,4,1)
my_graph.add_edge_weight(4,7,1)
my_graph.add_edge_weight(4,5,1)
my_graph.add_edge_weight(4,6,1)
my_graph.add_edge_weight(5,7,1)
my_graph.add_edge_weight(6,5,1)
my_graph.add_edge_weight(6,8,1)
my_graph.add_edge_weight(7,1,1)
my_graph.add_edge_weight(8,3,1)
my_graph.add_edge_weight(8,2,1)

#print(my_graph.compute_bellmanford(6,7))
print(my_graph.GloutonFas())
