import numpy as np


class Decoder():
    def __init__(self):
        pass

    def beam_search(self, mat, alphabet, blank_index, k):

        T = mat.shape[0]
        Probs = [{'': {'b': 1, 'nb': 0}}]
        # [{ beam: {'b': Pb[t], 'nb': Pnb[t]} }]

        for t in range(1, T + 1):
            next_probs = {beam + letter: {'b': 0, 'nb': 0} for beam in Probs[t - 1].keys() for letter in
                          alphabet + ['']}

            for beam in Probs[t - 1].keys():
                if beam != '':
                    next_probs[beam]['nb'] += Probs[t - 1][beam]['nb'] * mat[t - 1][alphabet.index(beam[-1])]
                next_probs[beam]['b'] += (Probs[t - 1][beam]['b'] + Probs[t - 1][beam]['nb']) * mat[t - 1][blank_index]

                for i in range(len(alphabet)):
                    new_beam = beam + alphabet[i]
                    if beam != '' and beam[-1] == alphabet[i]:
                        next_probs[new_beam]['nb'] += Probs[t - 1][beam]['b'] * mat[t - 1][i]
                    else:
                        next_probs[new_beam]['nb'] += (Probs[t - 1][beam]['b'] + Probs[t - 1][beam]['nb']) * mat[t - 1][
                            i]

            next_probs = dict(sorted(next_probs.items(), key=lambda d: d[1]['b'] + d[1]['nb'], reverse=True)[:k])
            Probs.append(next_probs)

        return Probs


D = Decoder()
mats = np.exp(np.load('PR_first70.npy'))
letters = []
fi = open("letters.lst", 'r', encoding='utf8')
for line in fi.readlines():
    letters.append(line[0])
mat = mats[-1]
print([letters[np.argmax(x)] for x in mat])

ans = D.beam_search(mat, letters, 1, 10)
for a in ans:
    print(sorted(a.items(), key=lambda d: d[1]['b'] + d[1]['nb'], reverse=True)[0][0])




#
#
#
#
#
#
#
#
#
#
#
#
#
