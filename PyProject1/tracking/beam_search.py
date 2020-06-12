import numpy as np

default_alphabet = [' ', '-', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с',
                    'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё']


def beam_search(mat: np.array, alphabet: list = default_alphabet, blank_index=-1, k=30, model=lambda x: 1):
    T = mat.shape[0]
    mat = np.exp(mat)
    Probs = [{'': {'b': 1, 'nb': 0}}]
    # [{ beam: {'b': Pb[t], 'nb': Pnb[t]} } for t]
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

        next_probs1 = dict(
            sorted(next_probs.items(), key=lambda d: model(d[0]) * (d[1]['b'] + d[1]['nb']), reverse=True)[:k])
        s = sum([v['b'] + v['nb'] for v in next_probs1.values()])
        next_probs = {k: {'nb': next_probs1[k]['nb'] / s, 'b': next_probs1[k]['b'] / s} for k in next_probs1.keys()}
        Probs.append(next_probs)

    return sorted(Probs[-1].items(), key=lambda x: x[1]['b'] + x[1]['nb'], reverse=True)[0][0]
