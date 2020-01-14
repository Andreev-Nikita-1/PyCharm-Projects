import numpy as np
import json
import Levenshtein
from tracking.beam_search import *
import os

service = 'ems'


def index_checking(numbers_str):
    if service == 'ems':
        return True
    else:
        index = numbers_str[:6]
        post_inds = np.load("post_indices.npy")
        return index in post_inds


def control_number_checking(numbers_str):
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


def numbers_norm(numbers_words):
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
            reduced1 += reducing([digit] + answer_num, expected_length=5 if service == 'ems' else 14)
            for i in range(len(answer_num)):
                reduced1 += reducing(answer_num[:i + 1] + [digit] + answer_num[i + 1:],
                                     expected_length=5 if service == 'ems' else 14)
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


# def por_answers(start=0, end=20):
#     for i in range(start, end):
#         v = np.load('data/Por_probs/' + str(i + 1) + '.npy')
#         print(numbers_norm(beam_search(v)))


def por():
    all = 0
    cor = 0
    len_ = 0
    len_cor = 0
    with open("data/por_keys.json", encoding='utf-8') as d:
        keys = json.loads(d.read())
    folders = os.listdir('data/Por_toloka')
    for folder in folders:
        files = os.listdir('data/Por_toloka/' + folder)
        correct = 0
        for file in files:
            v = np.load('data/Por_toloka/' + folder + '/' + file)
            id = file[:-4]
            beam = beam_search(v, k=1)

            ans = numbers_norm(beam.split())
            len_ += len(ans)
            if keys[id] in ans:
                correct += 1
                len_cor += len(ans)
            else:
                print('    ', beam, '\n     ', keys[id], len(ans), ans, id)

        print(correct, '/', len(files), '  :   ', folder)
        cor += correct
        all += len(files)
    print('all: ', cor, '/', all, '       len: ', len_ / all, '    len cor: ', len_cor / cor)


def ems():
    all = 0
    cor = 0
    len_ = 0
    lens_ = 0
    len_cor = 0
    lens_cor = 0
    cort = 0
    corn = 0
    corc = 0
    with open("data/ems_keys.json", encoding='utf-8') as d:
        keys = json.loads(d.read())
    folders = os.listdir('data/Ems_toloka')
    for folder in folders:
        files = os.listdir('data/Ems_toloka/' + folder)
        print('    ', folder)
        correct = 0
        for file in files:
            v = np.load('data/Ems_toloka/' + folder + '/' + file)
            id = file[:-4]
            beam = beam_search(v, k=5)
            # ans_lens = ems_norm(beam)
            ts, ns, cs = ems_norm(beam)
            # ans = ans_lens[0]
            # lens = ans_lens[1]
            # len_ += len(ans)
            # lens_ += lens
            real = keys[id]
            if real[:2] in ts and real[2:11] in ns and real[-2:] in cs:
                correct += 1

                # len_cor += len(ans)
                # lens_cor += lens
            else:
                print(len(ts), len(ns), len(cs))
                print(beam)
                if real[:2] not in ts:
                    cort += 1
                    print('t', end=" ")
                if real[2:11] not in ns:
                    print('n', end=" ")
                    corn += 1
                if real[-2:] not in cs:
                    print('c', end=" ")
                    corc += 1
                print(real, ts, ns, cs)
                # print('    ', beam, '\n     ', keys[id], lens, len(ans), ans, id)
        # print(correct, '/', len(files), '  :   ', folder)
        cor += correct
        all += len(files)
    # print('all: ', cor, '/', all, '       len: ', len_ / all, '    len cor: ', len_cor / cor, "   lens: ", lens_ / all,
    #       "   lens cor: ", lens_cor / all)
    print(cort, corn, corc, cor, ' / ', all)


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
    return [p[0] for p in pairs if Levenshtein.distance(p[1], s) - min_dist <= 0]


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
    return [p[0] for p in pairs if Levenshtein.distance(p[1], s) - min_dist <= 0 and p[0] in countries]


def ems_norm(beam):
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
    ts = type_norm(type_words)
    ns = numbers_norm(numbers_words)
    cs = country_norm(country_words)
    # return [t + n + c for t in ts for n in ns for c in cs], np.array([len(ts), len(ns), len(cs)])
    return ts, ns, cs


