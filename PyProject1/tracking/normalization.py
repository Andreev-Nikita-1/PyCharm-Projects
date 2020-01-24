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
    # return [t + n + c for t in ts for n in ns for c in cs], np.array([len(ts), len(ns), len(cs)])
    return ts, ns, cs


def control():
    correct = ["CA007433791SK(1).npy", "CA007433791SK.npy", "CA032156295RU.npy", "CA460681341AT(1).npy",
               "CA460681341AT.npy", "CB001324705RU.npy", "CC001848257AM(1).npy", "CC001848257AM.npy",
               "CC015274068ES.npy", "CC015308345ES(1).npy", "CC015308345ES.npy", "CC015308345IC(1).npy",
               "CC015308345IC.npy", "CC060510876IL.npy", "CC066204605IL.npy", "CD389708309JP.npy",
               "CD532874587NL(1).npy", "CD532874587NL.npy", "CE384308975BE(1).npy", "CE384308975BE.npy",
               "CF190999356DE(1).npy", "CF190999356DE.npy", "CG002327031IS(1).npy", "CG002327031IS(2).npy",
               "CG002327031IS.npy", "CG014604718DE.npy", "CG103316599LT(1).npy", "CG103316599LT.npy",
               "CH080033447US.npy", "CH081445532US.npy", "CH108815211AU(1).npy", "CH108815211AU.npy",
               "CJ003862203RU.npy", "CK068015119DE.npy", "CL001191142RU.npy", "CL056436176JP.npy",
               "CO821284605DE(1).npy", "CO821284605DE.npy", "CV033885375CZ(1).npy", "CV033885375CZ.npy",
               "CX427163343US.npy", "CY114763116US(1).npy", "CY114763116US.npy", "EE005074800RU.npy",
               "EE790557156TW(1).npy", "EE790557156TW.npy", "EF014887886RU(1).npy", "EF014887886RU(2).npy",
               "EF014887886RU.npy", "EH004852600US.npy", "EP056592600RU.npy", "EP073906229RU.npy",
               "EW001430412IT(1).npy", "EW001430412IT.npy", "EZ188779657US(2).npy", "EZ188779657US.npy",
               "LG618690024GB.npy", "LO122677706CN.npy", "LS824805509CH(2).npy", "LS824805509CH.npy",
               "LS890460115CH.npy", "LZ432379988US(1).npy", "LZ432379988US.npy", "LZ497696959IC(1).npy",
               "LZ497696959IC.npy", "RA019477640RU.npy", "RA253321807FI.npy", "RA259640517FI.npy", "RA612153932UA.npy",
               "RB012363705RU.npy", "RB601208140SG(1).npy", "RB601208140SG.npy", "RB791356155SG(2).npy",
               "RB791356155SG.npy", "RC092107717IT(1).npy", "RC092107717IT.npy", "RC727913178RG.npy",
               "RD009172569HK.npy", "RD213349651SE.npy", "RD331395160IN.npy", "RE540161296UA.npy", "RE684604472GR.npy",
               "RF020723158UA.npy", "RG973774831CN.npy", "RP019120314NL.npy", "RP900199158SG.npy", "RQ004050118RU.npy",
               "RQ020905632CY.npy", "RQ108337176UZ.npy", "RQ108337176UZ9(1).npy", "RR000396224GE.npy",
               "RR061424096BY.npy", "RR170116017TH.npy", "RR321763837PL.npy", "RS002497336MN.npy",
               "RU480548222HK(2).npy", "RU480548222HK.npy", "RV168615668CN.npy", "RV199755489CN(2).npy",
               "RV199755489CN.npy", "RX422387137CN.npy", "RY009826924HK(1).npy", "RY009826924HK(2).npy",
               "RY009826924HK.npy", "RZ022677306LV.npy", "UA789445529HK.npy", "UB082931076HK.npy", "UC460345146YP.npy",
               "UR521770422CN.npy", "VI629577860CN.npy", "ZA502560528LV(2).npy", "ZA502560528LV.npy",
               "ZA523577679LV.npy", "ZB020503497HK.npy", "ZB021478900HK(1).npy", "ZB021478900HK.npy",
               "ZC008498768HK.npy", "LL136789702CN.npy", "RP731131053CN.npy", "LL136789702CN(2).npy",
               "RP677676920CN.npy", "RA010477280JP.npy", "RP731131053CN(1).npy", "LO122677706CN(2).npy",
               "LC004077464CN.npy", "za669962194hk.npy", "UH879243837WS.npy", "za669962194hk(2).npy",
               "CT616989794CN(2).npy", "UH879243837WS(2).npy", "ZA123456789HK.npy", "CT616989794CN.npy"]
    folder = 'data/control_set'
    files = os.listdir(folder)
    out = open("data/control_out.txt", "w")
    tr, fs = 0, 0
    for file in set(files) - set(correct):
        real = file[:13].upper()
        nb = [int(s) for s in real[2:11]]
        if not control_number_ems(nb[:-1]) == nb[-1]:
            out.write('\"' + file + '\"' + ',')
            continue
        mat = np.load(folder + '/' + file)
        beam = beam_search(mat)
        letters = get_letters() + ['|']

        ts, ns, cs = ems_norm(beam)
        try:
            ans = real[:2] in ts and real[2:11] in ns and real[-2:] in cs
            # out.write(real + ' ' + ts[0] + ns[0] + cs[0] + '\n')
            if ans:
                out.write('\"' + file + '\"' + ',')
                tr += 1
            else:
                print()
                if real[:2] not in ts:
                    print('t', end=' ')
                if real[2:11] not in ns:
                    print('n', end=' ')
                if real[-2:] not in cs:
                    print('c')
                print("".join([letters[np.argmax(p)] for p in mat if letters[np.argmax(p)] != '']))
                print(beam)
                # print(ans, real)
                fs += 1
                print(real, ts, ns, cs)
        except:
            fs += 1
            print(real, ts, ns, cs)
    # print(tr / (tr + fs))

#
#
#


#
#
#
#
#
