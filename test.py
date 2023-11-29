import random

liste_edges = [tuple(random.sample([0,1,2,3,4,5,6,7],2)+[1]) for _ in range(13)]
print(liste_edges)