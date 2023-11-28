from graph import Graph

def main():
    g2 = Graph([0, 1, 2, 3, 4, 5, 6, 7])
    liste_arcs = [(0, 1, 1), (0, 2, 1), (2, 3, 1), (3, 4, 1), (3, 5, 1), (3, 6, 1), (4, 6, 1), (5, 4, 1),
                  (5, 7, 1)]
    g2.add_edges(liste_arcs)
    ordre = g2.glouton_fas()
    print(ordre)


if __name__ == '__main__':
    main()