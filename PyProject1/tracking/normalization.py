import numpy as np
import json
import Levenshtein
from .beam_search import *


def answer(beam):
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    answer_words = beam.split(' ')
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
    for i in range(start, end):
        v = np.load('data/Por_probs/' + str(i + 1) + '.npy')
        print(answer(beam_search(v)))
