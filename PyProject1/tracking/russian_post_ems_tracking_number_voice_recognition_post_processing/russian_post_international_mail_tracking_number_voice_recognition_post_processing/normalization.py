import json
import Levenshtein
from .beam_search import *
from .tracking_number_recognition import TrackingNumberRecognizer

letters_russian_pronunciation = json.load(open("letters_russian_pronunciation.json", encoding='utf-8'))
words_to_numbers = json.load(open("words_to_numbers.json.json", encoding='utf-8'))
post_indices = json.load(open("post_indices.json", 'r'))


class Normalization:
    """
    Class containing functions for converting track number into proper format
    """
    recognizer: TrackingNumberRecognizer

    def __init__(self, recognizer: TrackingNumberRecognizer):
        self.recognizer = recognizer

    def __index_checking(self, numbers_str: str) -> float:
        """
        Checks that post index in tracking number for "Russian Post" is correct
        """
        if self.recognizer.service == "International mail":
            return True
        elif self.recognizer.service == "Russian Post":
            index = numbers_str[:6]
            return index in post_indices
        else:
            raise ValueError('service must be "International mail" or "Russian Post"')

    def __control_number_checking(self, numbers_str: str) -> float:
        """
        Check that control number is correct
        """
        numbers = [int(s) for s in numbers_str]
        if self.recognizer.service == 'International mail':
            return self.__control_number_im(numbers[:-1]) == numbers[-1]
        elif self.recognizer.service == "Russian Post":
            return self.__control_number_por(numbers[:-1]) == numbers[-1]
        else:
            raise ValueError('service must be "International mail" or "Russian Post"')

    def __control_number_por(self, numbers: list) -> int:
        """
        Control number calculating algorithm for Russian Post taken from Wikipedia
        """
        sum = np.sum([3 * numbers[i] if i % 2 == 0 else numbers[i] for i in range(len(numbers))])
        result = 10 - sum % 10
        return result if result < 10 else 0

    def __control_number_im(self, numbers: list) -> int:
        """
        Control number calculating algorithm for International mail taken from Wikipedia
        """
        coeffs = np.array([8, 6, 4, 2, 3, 5, 9, 7])
        result = 11 - np.sum(coeffs * numbers) % 11
        if result == 10:
            result = 0
        elif result == 11:
            result = 5
        return result

    def __numbers_norm(self, numbers_words: list) -> list:
        """
        Returns list of strings, containing sequence of digits that matches the given words

        :param numbers_words: words, that correspond to digits pronounced, e.g. ["шестьдесят", "пять", "двадцать", "один"]
        :return: digits that could be pronounced, e.g. ["65201", "60521"] if length is supposed to be 5
        """
        all_words_nums = list(words_to_numbers.keys())
        nearest_numbers = [
            sorted(
                [
                    (
                        num,
                        Levenshtein.distance(num, word)
                    )
                    for num in all_words_nums
                ],
                key=lambda p: p[1]
            )[0]
            for word in numbers_words
        ]
        numbers_words = [p[0] for p in nearest_numbers if p[1] <= 1]
        numbers = [words_to_numbers[nw] for nw in numbers_words]

        reduced = self.__reducing(numbers, expected_length=9 if self.recognizer.service == 'im' else 14)

        # checking corectness
        reduced = [r for r in reduced if self.__control_number_checking(r) and self.__index_checking(r)]

        if not reduced:
            raise ValueError("Can't decode digits")
        return reduced

    def __reducing(self, numbers: list, expected_length: int) -> list:
        """
        Choose correct sequence of digits of given length
        """

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

    def __not_compatible_pairs(self) -> list:
        """
        Chooses pairs of english letters and their pronunciations, that unlikely to be said by one person
        """
        wrongs = [("C", "с"), ("C", "эс"), ("C", "сэ"), ("I", "и")]
        rights = [("E", "и"), ("I", "ай"), ("Y", "уай"), ("Y", "вай")]
        return [(x[0], x[1], y[0], y[1]) for x in rights for y in wrongs] + \
               [(x[0], x[1], y[0], y[1]) for x in wrongs for y in rights]

    def __two_letters_in_russian(self, AB: str) -> list:
        """
        Returns all probable pronunciations of given pair of letters
        """
        A, B = AB[0], AB[1]
        lrp = letters_russian_pronunciation
        ans = []
        if A == B:
            # the same two letters are unlikely to be pronounced in different ways
            for p in lrp[A + "_"]:
                ans.append(p + " " + p)
        else:
            ncp = self.__not_compatible_pairs()
            for p1 in lrp[A]:
                for p2 in lrp[B]:
                    # case when person pronounce letters together
                    ans.append(p1 + p2)
            for p1 in lrp[A + "_"]:
                for p2 in lrp[B + "_"]:
                    # case when person pronounce letters apart
                    if not (A, p1, B, p2) in ncp:
                        ans.append(p1 + p2)
                        ans.append(p1 + " " + p2)
        return ans

    def __probable_two_letters(self, beam_word: str, mat: np.array, prior_probs: dict) -> list:
        """
        Chooses the most probable pronunciations of two letters that corresponds to given matrix

        :param beam_word: word, received from beam search
        :param mat: 2-d matrix of probabilities that corresponds to two pronounced letters
        :param prior_probs: prior probabilities for pairs of letters
        :return: list of tuples (str, float) that corresponds to letters and probability
        """
        options = []
        AB_set = prior_probs.keys()
        for p in AB_set:
            for w in self.__two_letters_in_russian(p):
                options.append((p, w, Levenshtein.distance(w, beam_word)))

        # using all options counting ctc_prob can take long time
        options = sorted(options, key=lambda o: o[2])
        min_dist = options[0][2]
        options = [x for x in options if x[2] <= 2 + min_dist][:100]

        best = [x[0] for x in options]
        ctc_probs = np.array([ctc_prob(x[1], mat) for x in options])
        ctc_probs = ctc_probs / (ctc_probs.sum() + 0.1)

        # choosing best two letters in case of value ctc_prob * (1 - error_prob) + prior_prob * error_prob
        answers = np.array(
            list(sorted(
                [
                    (
                        ab,
                        ctc * (1 - self.recognizer.error_prob) + prior_probs[ab] * self.recognizer.error_prob
                    )
                    for ab, ctc in zip(best, ctc_probs)
                ],
                key=lambda x: x[1], reverse=True)))
        answers0 = [x[0] for x in answers]
        _, idx = np.unique(answers0, return_index=True)
        answers = list(answers[sorted(idx)])
        return answers

    def __parse_beam(self, beam: str) -> (list, list, list):
        """
        Splits string into three parts in terms of "International mail" tracking number format
        """
        words_nums = list(words_to_numbers.keys())
        beam_words = beam.split()
        closest_nums = [min([Levenshtein.distance(wn, w) for wn in words_nums]) for w in beam_words]
        almost_num = [dist <= 1 / 5 * len(w) for dist, w in zip(closest_nums, beam_words)]
        first_num_ind = almost_num.index(True)
        last_num_ind = almost_num[::-1].index(True)
        t_words = beam_words[:first_num_ind]
        n_words = beam_words[first_num_ind: len(beam_words) - last_num_ind]
        c_words = beam_words[-last_num_ind:]
        return t_words, n_words, c_words

    def __thresholds(self, mat: np.array, n_words: list) -> (int, int):
        """
        Splits matrix into three parts in terms of "International mail" tracking number format
        """
        first_num = n_words[0]
        last_num = n_words[-1]
        space_inds = [0] + [i for i in range(len(mat)) if
                            (self.recognizer.alphabet + ["|"])[np.argmax(mat[i])] == " "] + [len(mat) - 1]
        space_pairs = list(zip(space_inds, space_inds[1:]))
        tmoment, cmoment = 0, 0
        for i, j in space_pairs:
            if beam_search(mat[i:j], self.recognizer.alphabet).strip() == first_num:
                tmoment = i
                break
        for i, j in space_pairs[::-1]:
            if beam_search(mat[i:j], self.recognizer.alphabet).strip() == last_num:
                cmoment = j + 1
                break
        return tmoment, cmoment

    def __sort_best(self, ts: list, ns: list, cs: list) -> list:
        """
        Chooses most probable tracking numbers, gathering decoded three parts in terms of "International mail" tracking number format
        """
        pairs = [(t, c, t_prob * c_prob) for (t, t_prob) in ts for (c, c_prob) in cs]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return [t + ns[0] + c for (t, c, _) in pairs]

    def __im_norm(self, mat: np.array, string: str) -> list:
        """
        Decodes "International mail" tracking number

        :param mat: 2-d matrix of probabilities
        :param string: string received from beam search
        :return: list of most probable tracking numbers
        """

        t_words, n_words, c_words = self.__parse_beam(string)

        ns = self.__numbers_norm(n_words)

        tmoment, cmoment = self.__thresholds(mat, n_words)
        tmat = mat[:tmoment]
        cmat = mat[cmoment:]

        ts = self.__probable_two_letters(" ".join(t_words), tmat, self.recognizer.types_prior)
        cs = self.__probable_two_letters(" ".join(c_words), cmat, self.recognizer.countries_prior)

        return self.__sort_best(ts, ns, cs)

    def __por_norm(self, string: str) -> list:
        """
        Decodes "Russian post" tracking number

        :param mat: 2-d matrix of probabilities
        :param string: string received from beam search
        :return: list of most probable tracking numbers
        """
        return self.__numbers_norm(string.split())

    def norm(self, mat: np.array, string: str) -> list:
        """
        Decodes tracking number

        :param mat: 2-d matrix of probabilities
        :param string: string received from beam search
        :return: list of most probable tracking numbers
        """
        if self.recognizer.service == "International mail":
            return self.__im_norm(mat, string)
        elif self.recognizer.service == "Russian Post":
            return self.__im_norm(string)
        else:
            raise ValueError('service must be "International mail" or "Russian Post"')
