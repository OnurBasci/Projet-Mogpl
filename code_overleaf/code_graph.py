class Graph:
    def __init__(self, list_vertex : list):
        # liste des sommets
        self.list_vertex : list = list_vertex
        # liste des arcs
        self.list_edges : list = []
        # un ordre des sommets
        self.vertex_order : list = None
        # représentation du graph sous forme de listes d'adjacence
        self.graph : dict = {x: [] for x in self.list_vertex}
        # distances pour l'algorithm de Bellman-Ford
        self.distances : np.ndarray = np.full(len(self.list_vertex), np.inf)
        # arborescence des plus courts chemins
        self.paths : list = None  
        # nombre d'itérations pour l'algorithm de Bellman-Ford
        self.nb_iter : int = 0