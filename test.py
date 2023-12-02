import random
from graph import Graph

# 7532
my_graph = Graph([0,1,2,3,4,5,6,7])
liste_arcs = [(0,1,-9),(0,2,-8),(1,2,5),(2,3,3),(3,5,-5),(3,6,-1),(3,4,10),(4,6,-1),(5,4,-4),(5,7,3),(6,0,4),(7,1,2),(7,2,-5)]
my_graph.add_edges(liste_arcs)

print(my_graph.bellman_ford(0))


g2 = Graph([0,1,2,3,4,5])
liste_arcs = [(0,1,8),(0,5,10),(1,2,1),(2,5,-4),(2,3,-1),(3,4,-2),(4,5,1),(5,3,2)]
g2.add_edges(liste_arcs)
print(g2.bellman_ford(0))
