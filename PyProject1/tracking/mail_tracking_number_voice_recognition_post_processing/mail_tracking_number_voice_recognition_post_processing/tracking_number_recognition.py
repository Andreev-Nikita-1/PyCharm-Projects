import numpy as np
import json
import pkg_resources
from .beam_search import beam_search

res_dir = pkg_resources.resource_filename("mail_tracking_number_voice_recognition_post_processing", "resources")


class TrackingNumberRecognizer:
    """
    Class that allows to convert 2-d matrix of probabilities into tracking number format
    """
    service: str
    alphabet = [' ', '-', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с',
                'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'ё']
    error_prob: float
    countries_prior = json.load(open(res_dir + "/countries_prior.json", "r"))
    types_prior = json.load(open(res_dir + "/types_prior.json", "r"))

    def __init__(self,
                 service: str,
                 alphabet: list = None,
                 error_prob=0.2,
                 countries_prior: dict = None,
                 types_prior: dict = None):
        """
        Creates an instance of class that allows to convert 2-d matrix of probabilities into tracking number format.

        Format example for ``International mail``: ``RO260964943RU``

        Format example for ``Russian Post``: ``1421171600738``

        Matrix first dimension corresponds to time moment.

        Matrix second dimension corresponds to a letter, last index has to correspond to *blank* (see CTC).

        Matrix value is the probability of pronunciation letter at the moment.

        :param service: "International mail" or "Russian Post"
        :param alphabet: letters order in your model (russian only)
        :param error_prob: probability of your model to make a mistake
        :param countries_prior: prior probability for last two letters in case of "International mail" format, which corresponds to country. (selective distribution from data of real mailing is set by default)
        :param types_prior: prior probability for first two letters in case of "International mail" format, which corresponds to type. (selective distribution from data of real mailing is set by default)
        """
        self.service = service
        if alphabet is not None:
            self.alphabet = alphabet
        self.error_prob = error_prob
        if countries_prior is not None:
            self.countries_prior = countries_prior
        if types_prior is not None:
            self.types_prior = types_prior

    def convert(self, matrix: np.array) -> list:
        """
        :param matrix: 2-d matrix of probabilities, last letter is *blank*
        :return: tracking number
        """
        string = beam_search(matrix, self.alphabet)
        return Normalization(self).norm(matrix, string)


from .normalization import Normalization
