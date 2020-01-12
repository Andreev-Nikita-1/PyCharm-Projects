import numpy as np
import json
import Levenshtein
from tracking.beam_search import *
import os
import csv


def control_number_checking(numbers_str):
    numbers = [int(s) for s in numbers_str]
    return control_number(numbers[:-1]) == numbers[-1]


def control_number(numbers):
    sum = np.sum([3 * numbers[i] if i % 2 == 0 else numbers[i] for i in range(len(numbers))])
    result = 10 - sum % 10
    return result if result < 10 else 0


def answer(beam):
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    answer_words = beam.split()
    wn_sorted = [sorted([(wn, Levenshtein.distance(wn, w)) for wn in words_nums], key=lambda p: p[1]) for w in
                 answer_words]
    n = len(wn_sorted)

    def dist(inds):
        return sum([wn_sorted[i][int(inds[i])][1] for i in range(n)])

    current_inds = np.zeros(n)
    reduced = []
    k = 0
    while reduced == [] or k < 3:
        answer_num = [words_to_num[wn_sorted[i][int(current_inds[i])][0]] for i in range(n)]
        reduced1 = reducing(answer_num, 14)
        reduced += [r for r in reduced1 if control_number_checking(r)]
        inds_array = [current_inds + v for v in np.eye(n)]
        dists = [dist(inds) for inds in inds_array]
        i = np.argmin(dists)
        current_inds = inds_array[i]
        k += 1
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
    for i in range(start, end):
        v = np.load('data/Por_probs/' + str(i + 1) + '.npy')
        print(answer(beam_search(v)))


def por():
    all = 0
    cor = 0
    with open("data/por_keys.json", encoding='utf-8') as d:
        keys = json.loads(d.read())

    folders = os.listdir('data/Por_toloka')
    for folder in folders[1:]:
        files = os.listdir('data/Por_toloka/' + folder)
        correct = 0
        for file in files:
            v = np.load('data/Por_toloka/' + folder + '/' + file)
            id = file[:-4]
            beam = beam_search(v, k=1)
            try:
                ans = answer(beam)
                if keys[id] in ans:
                    correct += 1
                else:
                    print('    ', beam, '\n     ', ans, keys[id], id)
            except:
                print('   error', beam, keys[id], id)

        print(correct, '/', len(files), '  :   ', folder)
        cor += correct
        all += len(files)
    print('all: ', cor, '/', all)
