import numpy as np
import json
import Levenshtein
from sympy import besseli

from tracking.beam_search import *
import time


def index_checking(numbers_str, service='ems'):
    if service == 'ems':
        return True
    index = numbers_str[:6]
    post_inds = np.load("post_indices.npy")
    return index in post_inds


def control_number_checking(numbers_str, service='ems'):
    numbers = [int(s) for s in numbers_str]
    if service == 'ems':
        return control_number_ems(numbers[:-1]) == numbers[-1]
    else:
        return control_number_por(numbers[:-1]) == numbers[-1]


def control_number_por(numbers):
    sum = np.sum([3 * numbers[i] if i % 2 == 0 else numbers[i] for i in range(len(numbers))])
    result = 10 - sum % 10
    return result if result < 10 else 0


def control_number_ems(numbers):
    coeffs = np.array([8, 6, 4, 2, 3, 5, 9, 7])
    result = 11 - np.sum(coeffs * numbers) % 11
    if result == 10:
        result = 0
    elif result == 11:
        result = 5
    return result


def numbers_norm(numbers_words, service='ems'):
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    wn_sorted = [sorted([(wn, Levenshtein.distance(wn, w)) for wn in words_nums], key=lambda p: p[1]) for w in
                 numbers_words]
    wns = [p[0][0] for p in wn_sorted if p[0][1] <= 1]
    answer_num = [words_to_num[wn] for wn in wns]
    reduced = reducing(answer_num, expected_length=9 if service == 'ems' else 14)
    reduced1 = [r for r in reduced if control_number_checking(r) and index_checking(r)]
    if reduced1 == []:
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for digit in digits:
            reduced1 += reducing([digit] + answer_num, expected_length=9 if service == 'ems' else 14)
            for i in range(len(answer_num)):
                reduced1 += reducing(answer_num[:i + 1] + [digit] + answer_num[i + 1:],
                                     expected_length=9 if service == 'ems' else 14)
        reduced1 = [r for r in reduced1 if control_number_checking(r) and index_checking(r)]
        reduced1 = list(np.unique(reduced1))
    return reduced1


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
    subs = subsets(reducing_numbers, diff)
    return [reduce(numbers, r_ns + [0]) for r_ns in subs]


def type_norm(words):
    with open("letters_to_eng.json", encoding='utf-8') as f:
        lte = json.loads(f.read())
    s = "".join(words)
    types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
    pairs = []
    for l1 in types:
        for o1 in lte[l1]:
            for l2 in lte.keys():
                for o2 in lte[l2]:
                    pairs.append((l1 + l2, o1 + o2))
    pairs.sort(key=lambda x: Levenshtein.distance(x[1], s))
    min_dist = Levenshtein.distance(pairs[0][1], s)
    return list([p[1] for p in pairs if Levenshtein.distance(p[1], s) - min_dist <= 0])


def country_norm(words):
    with open("letters_to_eng.json", encoding='utf-8') as f:
        lte = json.loads(f.read())
    countries = np.load("country_keys.npy")
    s = "".join(words)
    pairs = []
    for l1 in lte.keys():
        for o1 in lte[l1]:
            for l2 in lte.keys():
                for o2 in lte[l2]:
                    pairs.append((l1 + l2, o1 + o2))
    pairs.sort(key=lambda x: Levenshtein.distance(x[1], s))
    min_dist = Levenshtein.distance(pairs[0][1], s)
    ans = [p[1] for p in pairs if
           Levenshtein.distance(p[1], s) - min_dist <= 0 and
           p[0] in countries]
    k = 0
    while ans == []:
        k += 1
        ans = [p[0] for p in pairs if
               Levenshtein.distance(p[1], s) - min_dist <= k and
               p[0] in countries]
    return list(np.unique(ans))


def not_compatible_pairs():
    wrongs = [("C", "с"), ("C", "эс"), ("C", "сэ"), ("I", "и")]
    rights = [("E", "и"), ("I", "ай"), ("Y", "уай"), ("Y", "вай")]
    return [(x[0], x[1], y[0], y[1]) for x in rights for y in wrongs] + [(x[0], x[1], y[0], y[1]) for
                                                                         x in wrongs for y in rights]


def two_letters_in_russian(AB):
    A, B = AB[0], AB[1]
    with open("letters_to_eng.json", encoding='utf-8') as f:
        lte = json.loads(f.read())
    ans = []
    if A == B:
        for p in lte[A + "_"]:
            ans.append(p + " " + p)
    else:
        ncp = not_compatible_pairs()
        for p1 in lte[A]:
            for p2 in lte[B]:
                ans.append(p1 + p2)
        for p1 in lte[A + "_"]:
            for p2 in lte[B + "_"]:
                if not (A, p1, B, p2) in ncp:
                    ans.append(p1 + " " + p2)
    return ans


def get_types_list():
    types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
    letters = [chr(ord('A') + i) for i in range(26)]
    return [A + B for A in types for B in letters]


def get_countries_list():
    return np.load("countries.npy")


def probable_two_letters(AB_list, beam_word, mat):
    if beam_word == "ру":
        return ["RU"], ["RU"]
    options = []
    for p in AB_list:
        for w in two_letters_in_russian(p):
            # print(p, w, Levenshtein.distance(w, beam_word))
            options.append((p, w, Levenshtein.distance(w, beam_word)))
    # print("bb ww", beam_word)
    best_options_levenshtein = sorted(options, key=lambda o: o[2])
    min_dist = best_options_levenshtein[0][2]
    best_options_levenshtein = [x for x in best_options_levenshtein if x[2] <= 2 + min_dist]
    best_options = np.array(
        [x[0] for x in
         sorted(best_options_levenshtein, key=lambda x: np.power(0, x[2]) * ctc_prob(x[1], mat), reverse=True)])
    _, idx = np.unique(best_options, return_index=True)
    return list(best_options[sorted(idx)]), [x[0] for x in best_options_levenshtein]


def ems_norm(mat):
    start = time.time()
    beam = beam_search(mat)
    beam_t = time.time() - start
    start = time.time()
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    beam_words = beam.split()
    closest_nums = [min([Levenshtein.distance(wn, w) for wn in words_nums]) for w in beam_words]
    almost_num = [dist <= 1 / 5 * len(w) for dist, w in zip(closest_nums, beam_words)]
    first_num_ind = almost_num.index(True)
    last_num_ind = almost_num[::-1].index(True)
    numbers_words = beam_words[first_num_ind: len(beam_words) - last_num_ind]
    nums_inds_t = time.time() - start
    start = time.time()
    type_words = beam_words[:first_num_ind]
    country_words = beam_words[-last_num_ind:]
    first_num = numbers_words[0]
    last_num = numbers_words[-1]
    space_inds = [0] + [i for i in range(len(mat)) if (get_letters() + ["|"])[np.argmax(mat[i])] == " "]
    space_pairs = list(zip(space_inds, space_inds[1:]))
    tmoment, cmoment = 0, 0
    for i, j in space_pairs:
        if beam_search(mat[i:j]).strip() == first_num:
            tmoment = i
            break
    for i, j in space_pairs[::-1]:
        if beam_search(mat[i:j]).strip() == last_num:
            cmoment = j + 1
            break
    trash_t = time.time() - start
    start = time.time()
    tmat = mat[:tmoment]
    cmat = mat[cmoment:]

    print(show(mat))
    print(" ".join(type_words))
    print(" ".join(country_words))
    print(show(tmat))
    print(show(cmat))

    ts, ts1 = probable_two_letters(get_types_list(), " ".join(type_words), tmat)
    cs, cs1 = probable_two_letters(get_countries_list(), " ".join(country_words), cmat)
    prob_t = time.time() - start
    start = time.time()
    ns = numbers_norm(numbers_words)
    prob_nums_t = time.time() - start

    print("beam:", beam_t, "num inds:", nums_inds_t, "trash:", trash_t, "prob AB:", prob_t, "prob nums:", prob_nums_t)

    return ts[:10], ns[:10], cs[:10], ts1[:10], cs1[:10]


def str_to_inds(st):
    letters = get_letters()
    return [letters.index(s) for s in st]


def show(matr):
    letters = get_letters() + ["|"]
    return ("".join([letters[np.argmax(p)] for p in matr if letters[np.argmax(p)] != '']))


# mat_ = np.load('test_mat.npy')


def ctc_prob(s, mat, blank_index=-1):
    s = str_to_inds(s)
    mat = np.exp(mat)
    l = len(s)
    n = mat.shape[0]
    Pnb = -np.ones(shape=[n, l])
    Pb = -np.ones(shape=[n, l])

    def step(i, j, t):
        if i == j == -1:
            return 1 if t == "b" else 0
        elif i == -1:
            return 0
        elif j == -1:
            return mat[i, blank_index] * step(i - 1, j, "b") if t == "b" else 0
        else:
            if t == "b":
                if Pb[i, j] == -1:
                    value = (step(i - 1, j, "b") + step(i - 1, j, "nb")) * mat[i, blank_index]
                    Pb[i, j] = value
                return Pb[i, j]
            elif t == 'nb':
                if Pnb[i, j] == -1:
                    if j == 0 or s[j] != s[j - 1]:
                        value = (step(i - 1, j - 1, "b") + step(i - 1, j - 1, "nb") + step(i - 1, j, "nb")) * mat[
                            i, s[j]]
                    else:
                        value = (step(i - 1, j - 1, "b") + step(i - 1, j, "nb")) * mat[i, s[j]]
                    Pnb[i, j] = value
                return Pnb[i, j]

    return step(n - 1, l - 1, "b") + step(n - 1, l - 1, "nb")
