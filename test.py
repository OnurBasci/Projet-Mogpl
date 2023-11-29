import random
from graph import Graph


my_graph = Graph([0,1,2,3,4,5,6,7])
liste_arcs = [(0,1,1),(0,2,1),(1,2,1),(2,3,1),(3,6,1),(3,4,1),(3,5,1),(4,6,1),(5,4,1),(5,7,1),(6,0,1),(7,2,1),(7,1,1)]
my_graph.add_edges(liste_arcs)


G1 = Graph.generate_graphs_with_random_weights(my_graph,1)[0]
Graph.show_graph(G1)
for i in range(10):
    random_order = Graph.generate_random_order(G1)
    print(random_order)
    bellman = G1.search_bellman_ford(0, random_order)
    print(f"nb iter = {bellman[2]}")


