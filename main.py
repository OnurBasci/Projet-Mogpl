import numpy as np
from graph import Graph

# Fonction à enlever une fois celle de la question 3 crée
def generate_graph_manually()-> list[Graph]:
    graphs = []
    w = np.random.randint(1,20,8)
    for _ in range(3):
        my_graph = Graph([0,1,2,3,4,5,6,7])
        liste_arcs = [
            (0,1,w[np.random.randint(1,8,1)]),(0,2,w[np.random.randint(1,8,1)]),    
            (1,2,w[np.random.randint(1,8,1)]),(2,3,w[np.random.randint(1,8,1)]),
            (3,6,w[np.random.randint(1,8,1)]),(3,4,w[np.random.randint(1,8,1)]),
            (3,5,w[np.random.randint(1,8,1)]),(4,6,w[np.random.randint(1,8,1)]),
            (5,4,w[np.random.randint(1,8,1)]),(5,7,w[np.random.randint(1,8,1)]),
            (6,0,w[np.random.randint(1,8,1)]),(7,2,w[np.random.randint(1,8,1)]),
            (7,1,w[np.random.randint(1,8,1)])
        ]
        my_graph.add_edges(liste_arcs)
        graphs.append(my_graph)
    return graphs

def main():
    g1,g2,g3 = generate_graph_manually()

    paths_1,dist_1,nb_iter_1 = g1.search_bellman_ford(0)
    paths_2,dist_2,nb_iter_2 = g2.search_bellman_ford(0)
    paths_3,dist_3,nb_iter_3 = g3.search_bellman_ford(0)

    union_T = Graph.unifiy_paths(paths_1,paths_2,paths_3,g1.list_vertex)
    union_T.show_graph_info()
    ordre = union_T.glouton_fas()
    print(ordre)

if __name__ == "__main__":
    main()