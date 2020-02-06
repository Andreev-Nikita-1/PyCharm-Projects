import numpy as np
import json
import Levenshtein
from tracking.beam_search import *
import os


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


def two_letters_in_russian(s):
    A, B = s[0], s[1]
    with open("letters_to_eng.json", encoding='utf-8') as f:
        lte = json.loads(f.read())
    ans = []
    for p1 in lte[A]:
        for p2 in lte[B]:
            ans.append(p1 + p2)
    return ans


def get_types_list():
    types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
    letters = [chr(ord('A') + i) for i in range(26)]
    return [A + B for A in types for B in letters]


def get_countries_list():
    return np.load("country_keys.npy")


def ems_norm(mat):
    beam = beam_search(mat)
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    answer_words = beam.split()
    wn_sorted = [sorted([(wn, Levenshtein.distance(wn, w)) for wn in words_nums], key=lambda p: p[1]) for w in
                 answer_words]
    ds = [x[0][1] <= 0 for x in wn_sorted]
    first_num = ds.index(True)
    last_num = ds[::-1].index(True)
    numbers_words = answer_words[first_num:len(ds) - last_num]
    type_words = answer_words[:first_num]
    country_words = answer_words[len(ds) - last_num:]
    type_moment = np.argmax([ctc_prob(" ".join(type_words), mat[:i]) for i in range(100)])
    country_moment = np.argmax([ctc_prob(" ".join(type_words), mat[-i:]) for i in range(100)])
    type_mat = mat[:type_moment+5]
    country_mat = mat[-country_moment-5:]
    number_mat = mat[type_moment:mat.shape[0] - country_moment]
    print(show(type_mat))
    print(show(country_mat))
    types = []
    for t in get_types_list():
        for w in two_letters_in_russian(t):
            types.append((ctc_prob(w, type_mat), t, w))
    countries = []
    for t in get_countries_list():
        for w in two_letters_in_russian(t):
            countries.append((ctc_prob(w, country_mat), t, w))
    type = max(types, key=lambda x: x[0])
    country = max(countries, key=lambda x: x[0])
    print(type[2])
    print(country[2])
    ns = numbers_norm(numbers_words)
    # numbers = [(ctc_prob(n, number_mat), n) for n in ns]
    # number = max(numbers, key=lambda x: x[0])[1]

    return type[1], ns[0], country[1]


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
            return 1 if t == "nb" else 0
        elif i == -1 or j == -1:
            return 0
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
