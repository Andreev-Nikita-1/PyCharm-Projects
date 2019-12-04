import numpy as np
# from tracking.features_extractor import *
import json
from nltk.util import ngrams


def beam_search(mat, alphabet, blank_index, k, model=lambda x: 1):
    T = mat.shape[0]
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

        next_probs = dict(
            sorted(next_probs.items(), key=lambda d: model(d[0]) * (d[1]['b'] + d[1]['nb']), reverse=True)[:k])
        s = sum([v['b'] + v['nb'] for v in next_probs.values()])
        next_probs = {k: {'nb': next_probs[k]['nb'] / s, 'b': next_probs[k]['b'] / s} for k in next_probs.keys()}
        Probs.append(next_probs)

    return Probs


def answer(probs):
    # for i in range(len(probs)):
    return sorted(probs[-1].items(), key=lambda x: x[1]['b'] + x[1]['nb'], reverse=True)[0][0]


def get_letters():
    letters = []
    fi = open("letters.lst", 'r', encoding='utf8')
    for line in fi.readlines():
        if line != '-\n':
            letters.append(line[0])
    return letters


with open("words_to_numbers.json", encoding='utf-8') as f:
    words_to_num = json.loads(f.read())
words0 = words_to_num.keys()
words = []
for w in words0:
    if words_to_num[w] < 10 and words_to_num[w] > 0:
        for _ in range(9):
            words.append(w)
    else:
        words.append(w)

# d = {}
# for l1 in letters:
#     for l2 in letters:
#         for l3 in letters:
#             s = 0
#             for w in words:
#                 inds = w.findall(l1 + l2 + l3)
m = 1
grams_2 = []
for w in words:
    grams_2.extend(list(ngrams(w + ' ', 2)))
grams_1 = []
for w in words:
    grams_1.extend(list(ngrams(w, 1)))

d = {g2: 0 for g2 in grams_2}
for g2 in grams_2:
    d[g2] += 1
s = {g1: 0 for g1 in grams_1}
for g1 in grams_1:
    s[g1] += 1
for k in d.keys():
    d[k] = (d[k] + m / 36) / s[(k[0],)] + m
s1 = sum(s.values())
for k in s.keys():
    s[k] /= (s1 + m)

# data = np.load("PR_first70.npy")


# for v in data:
#     print(answer(beam_search(np.exp(v), letters, -1, 15)))

# D = Decoder()
# input = np.exp((Waw_to_probs('data/Post_Russia_Recordings_wav/1.wav')))
# input = np.exp(Waw_to_probs('data/recordings/60_7.wav'))
# input = np.exp(Waw_to_probs('data/recordings/lasso.wav'))
# answer(D.beam_search(input, get_letters()[:-1], -1, 10))

def model(w):
    if w == '':
        return 0
    x, y = w[:-1], w[-1]
    if x == '':
        return 1
    elif x != '' and (x[-1], y) in d.keys():
        return d[(x[-1], y)]
    else:
        return m / (36 * m + s1)


letters = get_letters()[:-1]
mats = np.exp(np.load('PR_first70.npy'))
for mat in mats:
    inds = [0] + list(range(2, 36))
    mat = mat[:, inds]
    ans1 = beam_search(mat, get_letters()[:-1], -1, 10)
    ans2 = beam_search(mat, letters, -1, 10, model=model)
    print('-m', answer(ans1))
    print('+m', answer(ans2))

    # print([letters[np.argmax(x)] for x in mat])
    #
    # for a in ans:
    #     print(sorted(a.items(), key=lambda d: d[1]['b'] + d[1]['nb'], reverse=True)[0][0])
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
