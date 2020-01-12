import numpy as np


def get_letters():
    letters = []
    fi = open("letters.lst", 'r', encoding='utf8')
    for line in fi.readlines():
        letters.append(line[0])
    return letters[:-1]


def beam_search(mat, alphabet=get_letters(), blank_index=-1, k=30, model=lambda x: 1):
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


def control_number_checking(numbers_str, control_number):
    numbers = [int(s) for s in numbers_str]
    return control_number(numbers[:-1]) == numbers[-1]


def control_number_por(numbers):
    sum = np.sum([3 * numbers[i] if i % 2 == 0 else numbers[i] for i in range(len(numbers))])
    result = 10 - sum % 10
    return result if result < 10 else 0







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
#
