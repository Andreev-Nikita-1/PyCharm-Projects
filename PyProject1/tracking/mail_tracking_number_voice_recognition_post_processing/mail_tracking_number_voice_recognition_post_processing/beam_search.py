import numpy as np


def beam_search(mat: np.array, alphabet: list, k=30) -> str:
    """
    Performs *beam search* and returns the most probable string

    :param mat: 2-d matrix of probabilities, last letter is *blank*
    :param alphabet: order of letters
    :param k: number of beams on one step
    :return: string that matches to given matrix
    """
    T = mat.shape[0]
    mat = np.exp(mat)
    Probs = [{'': {'b': 1, 'nb': 0}}]
    # [{ beam: {'b': Pb[t], 'nb': Pnb[t]} } for t]
    for t in range(1, T + 1):
        next_probs = {beam + letter: {'b': 0, 'nb': 0}
                      for beam in Probs[t - 1].keys()
                      for letter in alphabet + ['']}

        for beam in Probs[t - 1].keys():
            if beam != '':
                next_probs[beam]['nb'] += Probs[t - 1][beam]['nb'] * mat[t - 1][alphabet.index(beam[-1])]
            next_probs[beam]['b'] += (Probs[t - 1][beam]['b'] + Probs[t - 1][beam]['nb']) * mat[t - 1][-1]

            for i in range(len(alphabet)):
                new_beam = beam + alphabet[i]
                if beam != '' and beam[-1] == alphabet[i]:
                    next_probs[new_beam]['nb'] += Probs[t - 1][beam]['b'] * mat[t - 1][i]
                else:
                    next_probs[new_beam]['nb'] += (Probs[t - 1][beam]['b'] + Probs[t - 1][beam]['nb']) * mat[t - 1][
                        i]

        next_probs = dict(
            sorted(
                next_probs.items(),
                key=lambda d: (d[1]['b'] + d[1]['nb']), reverse=True
            )[:k]
        )
        s = sum([v['b'] + v['nb'] for v in next_probs.values()])

        next_probs = {
            k: {
                'nb': next_probs[k]['nb'] / s,
                'b': next_probs[k]['b'] / s
            }
            for k in next_probs.keys()
        }
        Probs.append(next_probs)

    return sorted(Probs[-1].items(), key=lambda x: x[1]['b'] + x[1]['nb'], reverse=True)[0][0]


def ctc_prob(string: str, mat: np.array, alphabet: list) -> float:
    """
    Calculates the probability of given string to match given matrix
    """

    def str_to_inds(str):
        return [alphabet.index(s) for s in str]

    string = str_to_inds(string)
    mat = np.exp(mat)
    length = len(string)
    n = mat.shape[0]
    Pnb = -np.ones(shape=[n, length])
    Pb = -np.ones(shape=[n, length])

    def step(i, j, t):
        if i == j == -1:
            return 1 if t == "b" else 0
        elif i == -1:
            return 0
        elif j == -1:
            return mat[i, -1] * step(i - 1, j, "b") if t == "b" else 0
        else:
            if t == "b":
                if Pb[i, j] == -1:
                    value = (step(i - 1, j, "b") + step(i - 1, j, "nb")) * mat[i, -1]
                    Pb[i, j] = value
                return Pb[i, j]
            elif t == 'nb':
                if Pnb[i, j] == -1:
                    if j == 0 or string[j] != string[j - 1]:
                        value = (step(i - 1, j - 1, "b") + step(i - 1, j - 1, "nb") + step(i - 1, j, "nb")) * mat[
                            i, string[j]]
                    else:
                        value = (step(i - 1, j - 1, "b") + step(i - 1, j, "nb")) * mat[i, string[j]]
                    Pnb[i, j] = value
                return Pnb[i, j]

    return step(n - 1, length - 1, "b") + step(n - 1, length - 1, "nb")
