from graph import Graph


my_graph = Graph([0,1,2,3,4,5,6,7])

my_graph.add_edge_weight(0,1,1)
my_graph.add_edge_weight(0,2,1)
my_graph.add_edge_weight(1,2,1)
my_graph.add_edge_weight(2,3,1)
my_graph.add_edge_weight(3,6,1)
my_graph.add_edge_weight(3,4,1)
my_graph.add_edge_weight(3,5,1)
my_graph.add_edge_weight(4,6,1)
my_graph.add_edge_weight(5,4,1)
my_graph.add_edge_weight(5,7,1)
my_graph.add_edge_weight(6,0,1)
my_graph.add_edge_weight(7,2,1)
my_graph.add_edge_weight(7,1,1)

path,distance,nb_iter = my_graph.compute_bellmanford(5,0)

print("path: ", path)
print("distance: ", distance)
print("nb_iter: ", nb_iter)