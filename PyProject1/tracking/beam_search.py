import numpy as np


class Decoder():
    def __init__(self):
        pass

    def beam_search(self, mat, abet, k):
        T = mat.shape[0]
        beams = ['']
        Pnb = [[]]

        for t in range(T):


            for b in beams:
                pass
