import numpy as np
import json
import Levenshtein
from tracking.beam_search import *
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


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
    return list(np.unique([p[0] for p in pairs if Levenshtein.distance(p[1], s) - min_dist <= 0]))


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
    ans = [p[0] for p in pairs if
           Levenshtein.distance(p[1], s) - min_dist <= 0 and
           p[0] in countries]
    k = 0
    while ans == []:
        k += 1
        ans = [p[0] for p in pairs if
               Levenshtein.distance(p[1], s) - min_dist <= k and
               p[0] in countries]
    return list(np.unique(ans))


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
    return ts, ns, cs


def str_to_inds(st):
    letters = get_letters()
    return [letters.index(s) for s in st]


def show(matr):
    letters = get_letters()
    print("".join([letters[np.argmax(p)] for p in matr if letters[np.argmax(p)] != '']))


mat_ = np.load('test_mat.npy')


def ctc_loss(matr, st):
    inds = str_to_inds(st)
    graph = tf.Graph()
    mat1 = matr.reshape(-1, 1, matr.shape[1])
    with graph.as_default():
        labels = tf.sparse_placeholder(tf.int32, [1, len(st)])
        inputs = tf.placeholder(tf.float32, [matr.shape[0], 1, matr.shape[1]])
        sequence_length = tf.placeholder(tf.int32, [1])
        # labels_ = tf.sparse.from_dense(labels)
        # inputs_ = tf.convert_to_tensor(labels)
        # sequence_length_ = tf.convert_to_tensor(sequence_length)
        ctc_loss1 = tf.nn.ctc_loss(labels=labels, inputs=inputs, sequence_length=sequence_length)
    labels_ = tf.constant(np.array(inds))
    inputs_ = tf.constant(mat1)
    sequence_length_ = tf.constant(np.array([len(st)]))
    feed = {labels: labels_,
            inputs: inputs_,
            sequence_length: sequence_length_}
    with tf.Session() as sess:
        return sess.run([ctc_loss1], feed_dict=feed)


ctc_loss(mat_, 'а')
# x = tf.placeholder(tf.float32, [2, 2])
# y = tf.placeholder(tf.float32, [2, 3])
# z = tf.linalg.matmul(x, y)
# a = np.array([[0, 1], [1, 0]])
# a = tf.convert_to_tensor(np.array([[0, 1], [1, 0]]))
# b = tf.convert_to_tensor(np.array([[1, 2, 3], [5, 6, 7]]))
# b = np.array([[1, 2, 3], [5, 6, 7]])
# feed = {x: a, y: b}
# with tf.Session() as sess:
#     ans = sess.run([z], feed_dict=feed)
# print(ans)

# with
# (
#
#
#
