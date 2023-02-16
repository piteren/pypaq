import Levenshtein


# returns Levenshtein-distance of two strings
def lev_dist(source :str, target :str):
    return Levenshtein.distance(source, target)

# returns Levenshtein-distance of two lists (or strings)
def lev_distL(source :list or str, target :list or str):

    # same
    if len(source) == len(target):
        if not sum([1 for e in zip(source,target) if e[0] == e[1]]):
            return 0

    # prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for _ in range(tlen+1)] for _ in range(slen+1)]
    for i in range(slen+1): dist[i][0] = i
    for j in range(tlen+1): dist[0][j] = j

    # count distance
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i + 1][j + 1] = min(dist[i][j + 1] + 1,    # delete
                                     dist[i + 1][j] + 1,    # insert
                                     dist[i][j] + cost)     # substitute

    return dist[-1][-1]

# selects two furthest (with lev_dist) of three sentences
def two_most_distanced(sa :str,sb :str,sc :str):
    palev = [[lev_dist(*pair), pair] for pair in [[sa, sb], [sb, sc], [sc, sa]]]
    return sorted(palev)[-1][1]