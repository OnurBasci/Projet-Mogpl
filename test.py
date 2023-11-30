import random
from graph import Graph

# 7532
my_graph = Graph([0,1,2,3,4,5,6,7])
liste_arcs = [(0,1,-9),(0,2,-8),(1,2,5),(2,3,3),(3,5,-5),(3,6,-1),(3,4,10),(4,6,-1),(5,4,-4),(5,7,3),(6,0,4),(7,1,2),(7,2,-5)]
my_graph.add_edges(liste_arcs)

Graph.show_graph(my_graph)
path,dist,nb_iter,state = my_graph.search_bellman_ford(0,None)

print(f"Chemin : {path}")
print(f"Distance : {dist}")
print(f"Nombre d'itérations : {nb_iter}")

