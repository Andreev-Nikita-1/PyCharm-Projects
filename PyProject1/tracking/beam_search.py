import numpy as np
# from tracking.features_extractor import *
import json
# from nltk.util import ngrams
import Levenshtein


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

        next_probs1 = dict(
            sorted(next_probs.items(), key=lambda d: model(d[0]) * (d[1]['b'] + d[1]['nb']), reverse=True)[:k])
        s = sum([v['b'] + v['nb'] for v in next_probs1.values()])
        next_probs = {k: {'nb': next_probs1[k]['nb'] / s, 'b': next_probs1[k]['b'] / s} for k in next_probs1.keys()}
        Probs.append(next_probs)

    return Probs


def get_letters():
    letters = []
    fi = open("letters.lst", 'r', encoding='utf8')
    for line in fi.readlines():
        letters.append(line[0])
    return letters


def control_number_checking(numbers_str):
    numbers = [int(s) for s in numbers_str]
    return control_number(numbers[:-1]) == numbers[-1]


def control_number(numbers):
    sum = np.sum([3 * numbers[i] if i % 2 == 0 else numbers[i] for i in range(len(numbers))])
    result = 10 - sum % 10
    return result if result < 10 else 0


def answer(beams):
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    answer_words = (sorted(beams[-1].items(), key=lambda x: x[1]['b'] + x[1]['nb'], reverse=True)[0][0]).split(' ')
    wn_sorted = [sorted([(wn, Levenshtein.distance(wn, w)) for wn in words_nums], key=lambda p: p[1]) for w in
                 answer_words]
    n = len(wn_sorted)

    def dist(inds):
        return sum([wn_sorted[i][int(inds[i])][1] for i in range(n)])

    current_inds = np.zeros(n)
    reduced = []
    while reduced == []:
        answer_num = [words_to_num[wn_sorted[i][int(current_inds[i])][0]] for i in range(n)]
        reduced = reducing(answer_num, 14)
        reduced = [r for r in reduced if control_number_checking(r)]
        inds_array = [current_inds + v for v in np.eye(n)]
        dists = [dist(inds) for inds in inds_array]
        i = np.argmin(dists)
        current_inds = inds_array[i]
    return reduced


def reducing(numbers, expected_length):
    def reducing_number(n, m):
        if n > 9 >= m > 0:
            return 1
        elif n > 90 >= m > 9:
            return 2
        else:
            return 0

    def subsets(array, e_l):
        if array == []:
            if e_l == 0:
                return [[]]
            else:
                return []
        a = [[0] + tail for tail in subsets(array[1:], e_l)]
        b = []
        if array[0] > 0:
            b = [[array[0]] + tail for tail in subsets(array[1:], e_l - array[0])]
        return a + b

    def reduce(nums, r_ns):
        ans = []
        for i in range(len(nums)):
            ans.append(nums[i] // np.power(10, r_ns[i]))
        return ''.join([str(n) for n in ans])

    length = sum([len(str(n)) for n in numbers])
    diff = length - expected_length
    reducing_numbers = [reducing_number(n, m) for n, m in zip(numbers, numbers[1:])]
    return [reduce(numbers, r_ns + [0]) for r_ns in subsets(reducing_numbers, diff)]


def por_answers(start=0, end=20):
    letters = get_letters()[:-1]
    for i in range(start, end):
        v = np.load('data/Por_probs/' + str(i + 1) + '.npy')
        print(answer(beam_search(np.exp(v), letters, -1, 15)))

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
