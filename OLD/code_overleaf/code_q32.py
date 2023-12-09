@staticmethod
def _find_negative_cycle( path : list) -> int:
    """
        Fonction qui trouve un cycle à poids négatif
        :param path: chemin
        :return: nombre d'itérations
    """
    already_seen = {}
    index_cycle = -1
    for i,value in enumerate(path):
        if value in already_seen:
            index_cycle = i
            break
        already_seen[value] = i
    i_start_cycle = already_seen[path[index_cycle]]
    cycle = path[i_start_cycle:index_cycle]
    return cycle