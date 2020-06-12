import os
from numpy.distutils.log import good
# from mail_tracking_number_voice_recognition_post_processing import TrackingNumberRecognizer
from tracking.mail_tracking_number_voice_recognition_post_processing.mail_tracking_number_voice_recognition_post_processing.tracking_number_recognition import \
    TrackingNumberRecognizer

from tracking.normalization import *
import numpy as np
import copy

ws_c = np.zeros(254)
ws_t = np.zeros(254)
ts_c = np.zeros(7)
ts_t = np.zeros(7)


def rtest_por():
    rec = TrackingNumberRecognizer("International mail")
    dir = "data/control_set"
    cor = {0: 0, 1: 0, 2: 0}
    print()
    print()
    print("test---------------------------------------------------------------")
    all = len(os.listdir(dir))
    for i, file in enumerate(os.listdir(dir)):
        real = file[:13].upper()
        ans = rec.convert(np.load(dir + "/" + file))
        if real in ans:
            print((1 + ans.index(real)) * "*")
        else:
            print("XXXXXXXXXXXXXXXXXX")
        if i % 10 == 0:
            print((i + 1), " / ", all)

def rtest_lib():
    rec = TrackingNumberRecognizer("International mail")
    dir = "data/control_set"
    cor = {0: 0, 1: 0, 2: 0}
    print()
    print()
    print("test---------------------------------------------------------------")
    all = len(os.listdir(dir))
    for i, file in enumerate(os.listdir(dir)):
        real = file[:13].upper()
        ans = rec.convert(np.load(dir + "/" + file))
        if real in ans:
            print((1 + ans.index(real)) * "*")
        else:
            print("XXXXXXXXXXXXXXXXXX")
        if i % 10 == 0:
            print((i + 1), " / ", all)


# def get_country_prior():
#     return np.exp(ws_t) / np.exp(ws_t).sum()
#
#
# def get_types_prior():
#     return np.array([[x / 26] * 26 for x in np.exp(ts_t) / np.exp(ts_t).sum()]).reshape(-1)


def get_country_prior(l):
    d = json.load(open("count259.json", "r"))["c"]
    with open("countries.json", "r") as file:
        cs = json.load(file)
    probs = np.ones(len(cs)) * l
    for key in d.keys():
        ind = cs.index(key)
        probs[ind] += d[key]
    return probs / probs.sum()


def get_types_prior(l):
    d = json.load(open("count259.json", "r"))["t"]
    types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
    letters = [chr(ord('A') + i) for i in range(26)]
    cs = [A + B for A in types for B in letters]
    probs = np.ones(len(cs)) * l
    for key in d.keys():
        for i, type in enumerate(cs):
            if type[:1] == key:
                probs[i] += d[key] / 26
    return probs / probs.sum()


def generate_file(type_flag):
    while True:
        if type_flag:
            type = np.random.choice(get_types_list(), p=get_types_prior())[:1]
        else:
            country = np.random.choice(get_countries_list(), p=get_country_prior())
        dir1 = "data/EMS_toloka_all"
        dir2 = "data/control_set"
        dir3 = "data/control_set_259"
        good_files = []
        f = open("data/ems_keys.json", "r")
        ems_keys = json.load(f)
        for file in os.listdir(dir1):
            real = ems_keys[file[:-4]]
            if type_flag and real[:1] == type or not type_flag and real[-2:] == country:
                good_files.append(dir1 + "/" + file)
        for file in os.listdir(dir2):
            real = file[:13].upper()
            if type_flag and real[:1] == type or not type_flag and real[-2:] == country:
                good_files.append(dir2 + "/" + file)
        for file in os.listdir(dir3):
            real = file[:13].upper()
            if type_flag and real[:1] == type or not type_flag and real[-2:] == country:
                good_files.append(dir3 + "/" + file)
        if good_files != []:
            return np.random.choice(good_files)


def ctest():
    dir = "data/control_set"
    cor = {0: 0, 1: 0, 2: 0}
    print()
    print()
    print("test---------------------------------------------------------------")
    all = len(os.listdir(dir))
    for i, file in enumerate(os.listdir(dir)):
        real = file[:13].upper()
        ts, ns, cs = ems_norm(np.load(dir + "/" + file), dir + "/" + file)
        t = real[:2]
        n = real[2:11]
        c = real[-2:]
        if n in _f(ns) and t in _f([x[0] for x in ts]) and c in _f([x[0] for x in cs]):
            cor[0] += 1
        if n in _s(ns) and t in _s([x[0] for x in ts]) and c in _s([x[0] for x in cs]):
            cor[1] += 1
        if n in _t(ns) and t in _t([x[0] for x in ts]) and c in _t([x[0] for x in cs]):
            cor[2] += 1

        print("test----" + "   1: {},   2: {},   3: {}".format(cor[0] / (i + 1), cor[1] / (i + 1), cor[2] / (i + 1)))
        print((i + 1), "/", all)

    print("----------------------------------------------------------------------")
    print()
    print()

    log = open("log.txt", "a")
    log.write("=================" + "\n\n")
    log.write("cor      " + str(cor[0] / all) + "   " + str(cor[1] / all) + "    " + str(cor[2] / all) + "\n")
    log.write("error=   " + str(error_prob))
    log.write(str(get_country_prior()) + "\n")
    log.write(str(get_types_prior()) + "\n\n")


def learning():
    global ws_c
    global ws_t
    global ts_c
    global ts_t

    update_period = 5
    test_period = 50
    lr = 2
    lr1 = 2
    for g in range(2000):
        print(g)
        print(ws_c[:20])
        print(ts_c)
        file = generate_file(True)
        if file[5] == 'E':
            f = open("data/ems_keys.json", "r")
            ems_keys = json.load(f)
            real = ems_keys[file[20:-4]]
        elif file[16] == '_':
            real = file[21:34].upper()
        else:
            real = file[17:30].upper()

        real_t = real[:2]
        real_c = real[-2:]

        ts, ns, cs = ems_norm(np.load(file), file)

        coeff = error_prob / (float(ts[0][1]) + 0.1)
        index = ['R', 'L', 'V', 'C', 'E', 'U', 'Z'].index(real_t[:1])

        vector = -np.exp(ts_c[index]) * np.exp(ts_c) / np.exp(ts_c).sum() ** 2
        vector[index] += np.exp(ts_c[index]) / np.exp(ts_c).sum()

        ts_c += vector * coeff * lr1

        file = generate_file(False)
        if file[5] == 'E':
            f = open("data/ems_keys.json", "r")
            ems_keys = json.load(f)
            print(file)
            real = ems_keys[file[20:-4]]
        elif file[16] == '_':
            real = file[21:34].upper()
        else:
            real = file[17:30].upper()

        real_t = real[:2]
        real_c = real[-2:]

        ts, ns, cs = ems_norm(np.load(file), file)

        coeff = error_prob / (float(cs[0][1]) + 0.1)
        index = get_countries_list().index(real_c)

        vector = -np.exp(ws_c[index]) * np.exp(ws_c) / np.exp(ws_c).sum() ** 2
        vector[index] += np.exp(ws_c[index]) / np.exp(ws_c).sum()

        ws_c += vector * coeff * lr

        if (g % update_period == 0):
            ws_t = copy.deepcopy(ws_c)
            ts_t = copy.deepcopy(ts_c)

        if (g % test_period == 0 and g != 0):
            ctest()


def collect_259():
    dir = "data/control_set_259"
    f = open("count259.json", "w")
    d: dict = {"c": {}, "t": {}}
    for i, file in enumerate(os.listdir(dir)):
        real = file[:13].upper()
        t = real[:1]
        n = real[2:11]
        c = real[-2:]

        if c not in d["c"].keys():
            d["c"][c] = 1
        else:
            d["c"][c] += 1

        if not t in d["t"].keys():
            d["t"][t] = 1
        else:
            d["t"][t] += 1

    json.dump(d, f)


def ctest_259():
    dir = "data/control_set_259"
    cor = {0: 0, 1: 0, 2: 0}
    print()
    print()
    print("test---------------------------------------------------------------")
    all = len(os.listdir(dir))
    for i, file in enumerate(os.listdir(dir)):
        real = file[:13].upper()
        ts, ns, cs = ems_norm(np.load(dir + "/" + file), dir + "/" + file)
        t = real[:2]
        n = real[2:11]
        c = real[-2:]
        if n in _f(ns) and t in _f([x[0] for x in ts]) and c in _f([x[0] for x in cs]):
            cor[0] += 1
        if n in _s(ns) and t in _s([x[0] for x in ts]) and c in _s([x[0] for x in cs]):
            cor[1] += 1
        if n in _t(ns) and t in _t([x[0] for x in ts]) and c in _t([x[0] for x in cs]):
            cor[2] += 1

        print("test----" + "   1: {},   2: {},   3: {}".format(cor[0] / (i + 1), cor[1] / (i + 1), cor[2] / (i + 1)))
        print((i + 1), "/", all)

    print("----------------------------------------------------------------------")
    print()
    print()


def ebtest():
    dir = "data/control_set"
    res = {}
    all = len(os.listdir(dir))
    ar = np.zeros((len(errors), len(bias)))
    for i, file in enumerate(os.listdir(dir)):
        real = file[:13].upper()
        ts_, ns, cs_ = ems_norm(np.load(dir + "/" + file), dir + "/" + file)
        t = real[:2]
        n = real[2:11]
        c = real[-2:]
        for e in errors:
            for b in bias:
                if str((e, b)) not in res.keys():
                    res[str((e, b))] = {0: 0, 1: 0, 2: 0}
                ts = ts_[(e, b)]
                cs = cs_[(e, b)]
                cor = res[str((e, b))]
                if n in _f(ns) and t in _f([x[0] for x in ts]) and c in _f([x[0] for x in cs]):
                    cor[0] += 1
                if n in _s(ns) and t in _s([x[0] for x in ts]) and c in _s([x[0] for x in cs]):
                    cor[1] += 1
                if n in _t(ns) and t in _t([x[0] for x in ts]) and c in _t([x[0] for x in cs]):
                    cor[2] += 1
        print((i + 1), "/", all)

    for i, e in enumerate(errors):
        for j, b in enumerate(bias):
            ar[i, j] = res[str((e, b))][0] / all

    json.dump(res, open("ebres.json", "w"))
    np.save("ebar.npy", ar)


x = np.load("ebar.npy")


def pre_search():
    dir1 = "data/EMS_toloka_all"
    dir2 = "data/control_set"
    dir3 = "data/control_set_259"
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    files3 = os.listdir(dir3)
    f = open("data/ems_keys.json", "r")
    ems_keys = json.load(f)
    sum = len(files1) + len(files2) + len(files3)
    k = 1
    res = open("beams.json", "a")

    for file in os.listdir(dir1):
        print(k, " / ", sum)
        k += 1
        real = ems_keys[file[:-4]]
        beam = beam_search(np.load(dir1 + "/" + file), k=5)
        print(real + "   " + beam)
        res.write('"' + dir1 + "/" + file + '"' + '"' + beam + '"' + "\n")

    for file in os.listdir(dir2):
        print(k, " / ", sum)
        k += 1
        real = file[:13]
        beam = beam_search(np.load(dir2 + "/" + file), k=5)
        print(real + "   " + beam)
        res.write('"' + dir2 + "/" + file + '"' + '"' + beam + '"' + "\n")

    for file in os.listdir(dir3):
        print(k, " / ", sum)
        k += 1
        real = file[:13]
        beam = beam_search(np.load(dir3 + "/" + file), k=5)
        print(real + "   " + beam)
        res.write('"' + dir3 + "/" + file + '"' + '"' + beam + '"' + "\n")


def _f(list):
    if len(list) > 0:
        return list[:1]
    else:
        return []


def _s(list):
    if len(list) > 1:
        return list[:2]
    else:
        return _f(list)


def _t(list):
    if len(list) > 2:
        return list[:3]
    else:
        return _s(list)


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


cr = ["CA032156295RU.npy", "CA460681341AT.npy", "CC001848257AM.npy", "CC015274068ES.npy", "CC015308345IC.npy",
      "CC066204605IL.npy", "CD389708309JP.npy", "CD532874587NL(1).npy", "CD532874587NL.npy", "CE384308975BE(1).npy",
      "CE384308975BE.npy", "CF190999356DE(1).npy", "CG002327031IS(2).npy", "CG103316599LT(1).npy", "CG103316599LT.npy",
      "CJ003862203RU.npy", "CK068015119DE.npy", "CL001191142RU.npy", "CO821284605DE(1).npy", "CO821284605DE.npy",
      "CV033885375CZ(1).npy", "CV033885375CZ.npy", "CX427163343US.npy", "CY114763116US(1).npy", "EE005074800RU.npy",
      "EE790557156TW(1).npy", "EF014887886RU(2).npy", "EF014887886RU.npy", "EH004852600US.npy", "EP056592600RU.npy",
      "EP073906229RU.npy", "EW001430412IT(1).npy", "EW001430412IT.npy", "EZ188779657US(2).npy", "EZ188779657US.npy",
      "LO122677706CN.npy", "LZ432379988US(1).npy", "LZ432379988US.npy", "LZ497696959IC(1).npy", "LZ497696959IC.npy",
      "RA010477280JP.npy", "RA019477640RU.npy", "RA253321807FI.npy", "RA612153932UA.npy", "RB012363705RU.npy",
      "RC092107717IT(1).npy", "RC727913178RG.npy", "RD009172569HK.npy", "RD213349651SE.npy", "RD331395160IN.npy",
      "RE540161296UA.npy", "RE684604472GR.npy", "RF020723158UA.npy", "RP019120314NL.npy", "RQ004050118RU.npy",
      "RQ020905632CY.npy", "RQ108337176UZ9(1).npy", "RR000396224GE.npy", "RR061424096BY.npy", "RR321763837PL.npy",
      "RU480548222HK(2).npy", "RU480548222HK.npy", "RV168615668CN.npy", "RV199755489CN(2).npy", "RV199755489CN.npy",
      "RX422387137CN.npy", "RY009826924HK(1).npy", "RY009826924HK(2).npy", "RY009826924HK.npy", "RZ022677306LV.npy",
      "UA789445529HK.npy", "UB082931076HK.npy", "UR521770422CN.npy", "ZA502560528LV(2).npy", "ZA523577679LV.npy",
      "za669962194hk.npy", "ZB020503497HK.npy", "ZB021478900HK(1).npy", "ZB021478900HK.npy", "ZC008498768HK.npy"]


class Checker:
    def __init__(self, n):
        self.n = n
        self.all = np.zeros(n)
        self.trues = np.zeros(n)
        # self.trues1 = np.zeros(n)
        # self.trues2 = np.zeros(n)
        # self.trues3 = np.zeros(n)
        # self.trues4 = np.zeros(n)
        # self.trues9 = np.zeros(n)

    def check(self, real, res, k):
        return real[:2] in res.ts[:k + 1] and real[2:11] in res.ns and real[-2:] in res.cs[:k + 1]

    def update(self, real, results):
        self.all += 1
        self.trues += np.array([self.check(real, r, 0) for r in results])
        # self.trues1 += np.array([self.check(real, r, 1) for r in results])
        # self.trues2 += np.array([self.check(real, r, 2) for r in results])
        # self.trues3 += np.array([self.check(real, r, 3) for r in results])
        # self.trues4 += np.array([self.check(real, r, 4) for r in results])
        # self.trues9 += np.array([self.check(real, r, 9) for r in results])
        print("----", self.all[0])
        names = [r.name for r in results]
        ress = self.trues / self.all
        print(sorted(zip(names, ress), key=lambda x: x[1], reverse=True))

        # print("  ".join([str(r) for r in self.trues / self.all]))
        # print("s1", "  ".join([str(r) for r in self.trues1 / self.all]))
        # print("s2", "  ".join([str(r) for r in self.trues2 / self.all]))
        # print("s3", "  ".join([str(r) for r in self.trues3 / self.all]))
        # print("s4", "  ".join([str(r) for r in self.trues4 / self.all]))
        # print("s9", "  ".join([str(r) for r in self.trues9 / self.all]))


def control():
    folder = 'data/control_set'
    # folder = 'data/Ems_toloka_all'
    files = os.listdir(folder)
    # np.shuffle(files)
    out = open("data/control_out.txt", "w")
    with open("data/ems_keys.json", encoding='utf-8') as d:
        keys = json.loads(d.read())
    checker = Checker(99)
    for i, file in enumerate(files):
        # if file in cr:
        #     continue
        real = file[:13].upper()
        # real = keys[file[:-4]]
        print(file, real)
        nb = [int(s) for s in real[2:11]]
        if not control_number_ems(nb[:-1]) == nb[-1]:
            out.write('\"' + file + '\"' + ',')
            print(file, "    number problems")
            continue
        mat = np.load(folder + '/' + file)
        checker.update(real, list(ems_norm(mat)))
        # print(ts, ns, cs)
        # ans = real[:2] in ts and real[2:11] in ns and real[-2:] in cs
        # if ans:
        #     # out.write('\"' + file + '\"' + ',')
        #     print("---correct---")
        #     i1 = ts.index(real[:2])
        #     i2 = ns.index(real[2:11])
        #     i3 = cs.index(real[-2:])
        #     print(i1, ts[:i1 + 1], "  ", i2, ns[:i2 + 1], "  ", i3, cs[:i3 + 1])
        #     tr += 1
        #     if i1 == i2 == i3 == 0:
        #         tr1 += 1
        #         out.write('\"' + file + '\"' + ',')
        #         print(r"\\\super correct!///")
        #     if i1 <= 1 and i3 <= 1:
        #         tr_1 += 1
        #     if i1 <= 2 and i3 <= 2:
        #         tr_2 += 1
        #     if i1 <= 3 and i3 <= 3:
        #         tr_3 += 1
        #     if i1 <= 4 and i3 <= 4:
        #         tr_4 += 1
        #
        # else:
        #     if real[:2] not in ts:
        #         print('t', end=' ')
        #     if real[2:11] not in ns:
        #         print('n', end=' ')
        #     if real[-2:] not in cs:
        #         print('c', end=' ')
        #     print("#$&#?@#*WRONG*#?@$#%$#  ", file, "     ", ts, ns, cs)
        # print("---------------------------------------------------")
        # print(i, "/", len(files), "    ", end="")
        # print("cor={}   {}".format(tr, tr / (i + 1)), "     sup cor={}   {}".format(tr1, tr1 / (i + 1)))
        # print("cor1={}   {}".format(tr_1, tr_1 / (i + 1)))
        # print("cor2={}   {}".format(tr_2, tr_2 / (i + 1)))
        # print("cor3={}   {}".format(tr_3, tr_3 / (i + 1)))
        # print("cor4={}   {}".format(tr_4, tr_4 / (i + 1)))


def ccc():
    cs = np.load("countries.npy")
    ts = {c: 0 for c in cs}
    folder = 'data/control_set'
    # folder = 'data/Ems_toloka_all'
    files = os.listdir(folder)
    with open("data/ems_keys.json", encoding='utf-8') as d:
        keys = json.loads(d.read())
    for i, file in enumerate(files):
        real = file[:13].upper()
        # real = keys[file[:-4]]
        ts[real[-2:]] += 1
    for c in cs:
        print("\"", c, "\":", ts[c], ",", end="", sep="")


def ccc():
    types = ['R', 'L', 'V', 'C', 'E', 'U', 'Z']
    ts = {c: 0 for c in types}
    folder = 'data/control_set'
    # folder = 'data/Ems_toloka_all'
    files = os.listdir(folder)
    for i, file in enumerate(files):
        real = file[:13].upper()
        # real = keys[file[:-4]]
        ts[real[:1]] += 1
    for c in types:
        print("\"", c, "\":", ts[c], ",", end="", sep="")

# control()

# aaa = np.load("test.npy")
#
# x=0
