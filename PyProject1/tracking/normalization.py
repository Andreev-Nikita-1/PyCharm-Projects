import json
import Levenshtein

from tracking.beam_search import *
from tracking.checking import *

beams_number = 5


def get_types_list():
    types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
    letters = [chr(ord('A') + i) for i in range(26)]
    return [A + B for A in types for B in letters]


def get_countries_list():
    with open("countries.json", "r") as file:
        return json.load(file)


#
# def get_country_prior():
#     with open("countries_prior.json", "r") as file:
#         return json.load(file)
#
#
# def get_types_prior():
#     types_aprior = {"R": 46, "L": 16, "V": 1, "C": 45, "E": 14, "U": 4, "Z": 11}
#     types_aprior = {t: types_aprior[t[0]] / 26 for t in get_types_list()}
#     return types_aprior


def get_error_prob():
    return 0.1


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
    reduced = [r for r in reduced if control_number_checking(r, service) and index_checking(r, service)]
    # if reduced1 == []:
    #     digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     for digit in digits:
    #         reduced1 += reducing([digit] + answer_num, expected_length=9 if service == 'ems' else 14)
    #         for i in range(len(answer_num)):
    #             reduced1 += reducing(answer_num[:i + 1] + [digit] + answer_num[i + 1:],
    #                                  expected_length=9 if service == 'ems' else 14)
    #     reduced1 = [r for r in reduced1 if control_number_checking(r) and index_checking(r)]
    #     reduced1 = list(np.unique(reduced1))
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
        if not array:
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


def not_compatible_pairs():
    wrongs = [("C", "с"), ("C", "эс"), ("C", "сэ"), ("I", "и")]
    rights = [("E", "и"), ("I", "ай"), ("Y", "уай"), ("Y", "вай")]
    return [(x[0], x[1], y[0], y[1]) for x in rights for y in wrongs] + \
           [(x[0], x[1], y[0], y[1]) for x in wrongs for y in rights]


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
                    ans.append(p1 + p2)
                    ans.append(p1 + " " + p2)
    return ans


error_prob = 0.01

errors = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
bias = [0.001, 0.01, 0.05, 0.1, 1, 5, 10, 50, 100]


def probable_two_letters(AB_list: list, beam_word: str, mat: np.array, pair_prior) -> list:
    options = []
    for p in AB_list:
        for w in two_letters_in_russian(p):
            options.append((p, w, Levenshtein.distance(w, beam_word)))

    # using all options counting ctc_prob can take long time
    options = sorted(options, key=lambda o: o[2])
    min_dist = options[0][2]
    options = [x for x in options if x[2] <= 2 + min_dist][:100]

    best = [x[0] for x in options]
    ctc_probs = np.array([ctc_prob(x[1], mat) for x in options])
    ctc_probs = ctc_probs / (ctc_probs.sum() + 0.1)

    # choosing best two letters in case of value ctc_prob * (1 - error_prob) + prior_prob * error_prob
    # prior_dict = dict(zip(AB_list, pair_prior))
    # answers = np.array(
    #     list(sorted([(ab, ctc * (1 - error_prob) + prior_dict[ab] * error_prob) for ab, ctc in zip(best, ctc_probs)],
    #                 key=lambda x: x[1], reverse=True)))
    # answers0 = [x[0] for x in answers]
    # _, idx = np.unique(answers0, return_index=True)
    # answers = list(answers[sorted(idx)])
    # return answers

    res = {}

    for e in errors:
        for b in bias:
            prior_dict = dict(zip(AB_list, pair_prior(b)))
            answers = np.array(
                list(sorted(
                    [(ab, ctc * (1 - e) + prior_dict[ab] * e) for ab, ctc in zip(best, ctc_probs)],
                    key=lambda x: x[1], reverse=True)))
            answers0 = [x[0] for x in answers]
            _, idx = np.unique(answers0, return_index=True)
            answers = list(answers[sorted(idx)])
            res[(e, b)] = answers
    return res


def parse_beam(beam: str):
    with open("words_to_numbers.json", encoding='utf-8') as f:
        words_to_num = json.loads(f.read())
    words_nums = list(words_to_num.keys())
    beam_words = beam.split()
    closest_nums = [min([Levenshtein.distance(wn, w) for wn in words_nums]) for w in beam_words]
    almost_num = [dist <= 1 / 5 * len(w) for dist, w in zip(closest_nums, beam_words)]
    first_num_ind = almost_num.index(True)
    last_num_ind = almost_num[::-1].index(True)
    t_words = beam_words[:first_num_ind]
    n_words = beam_words[first_num_ind: len(beam_words) - last_num_ind]
    c_words = beam_words[-last_num_ind:]
    return t_words, n_words, c_words


def thresholds(mat: np.array, n_words: list):
    first_num = n_words[0]
    last_num = n_words[-1]
    space_inds = [0] + [i for i in range(len(mat)) if (get_letters() + ["|"])[np.argmax(mat[i])] == " "] + [
        len(mat) - 1]
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
    return tmoment, cmoment


# beams = json.load(open("beams.json", "r", encoding="utf-"))


def ems_norm(mat: np.array, file="") -> (list, list, list):
    # if file == "":
    t_words, n_words, c_words = parse_beam(beam_search(mat, k=beams_number))
    # else:
    #     t_words, n_words, c_words = parse_beam(beams[file])

    ns = numbers_norm(n_words)

    tmoment, cmoment = thresholds(mat, n_words)
    tmat = mat[:tmoment]
    cmat = mat[cmoment:]

    ts = probable_two_letters(get_types_list(), " ".join(t_words), tmat, get_types_prior)
    cs = probable_two_letters(get_countries_list(), " ".join(c_words), cmat, get_country_prior)

    return ts, ns, cs


def ctc_prob(s: str, mat: np.array, blank_index=-1):
    def str_to_inds(st):
        letters = get_letters()
        return [letters.index(s) for s in st]

    s = str_to_inds(s)
    mat = np.exp(mat)
    length = len(s)
    n = mat.shape[0]
    Pnb = -np.ones(shape=[n, length])
    Pb = -np.ones(shape=[n, length])

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

    return step(n - 1, length - 1, "b") + step(n - 1, length - 1, "nb")
