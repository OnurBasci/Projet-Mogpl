import random
from graph import Graph

# 7532
my_graph = Graph([0,1,2,3,4,5,6,7])
liste_arcs = [(0,1,1),(1,3,1),(3,4,1)]
my_graph.add_edges(liste_arcs)
print(my_graph.bfs(0))
