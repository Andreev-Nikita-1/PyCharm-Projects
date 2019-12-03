import numpy as np
import scipy
import scipy.io.wavfile
import json
import onnx
import onnxruntime
import kaldi_features


class KaldiFeaturesExtractor:

    def __init__(self, params: dict):
        self.__caldi_features_calcer = kaldi_features.KaldiFeaturesCalcer(params)

    def extract(self, path):
        wave = scipy.io.wavfile.read(path)
        return self.__caldi_features_calcer.compute(wave[1].astype(np.float16))


class Network:
    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession("quartznet.onnx")
        for nn_input in self.onnx_session.get_inputs():
            print(nn_input)

    def apply(self, features):
        ort_outs = np.array(self.onnx_session.run(None, {"features": features}))
        return ort_outs.reshape(features.shape[0], -1, 36)


def Waw_to_probs(filename, configfile='features_config.json'):
    with open(configfile) as f:
        feature_calcer_params = json.loads(f.read())
    KFE = KaldiFeaturesExtractor(feature_calcer_params)
    res = KFE.extract(filename)
    l = 16 * np.ceil(res.shape[0] / 16)
    res = np.concatenate([res, np.zeros(shape=(int(l) - res.shape[0], 80), dtype=np.float32)], axis=0)
    tensor = np.array(res.T.reshape(1, 80, -1))
    Nw = Network()
    return Nw.apply(tensor).reshape(-1, 36)


def get_letters():
    letters = []
    fi = open("letters.lst", 'r', encoding='utf8')
    for line in fi.readlines():
        letters.append(line[0])
    return letters


def show(result):
    letters = get_letters()
    print("".join([letters[np.argmax(p)] for p in result if letters[np.argmax(p)] != '']))

# l = 896
# tensor = []
# for i in range(51, 71):
#     result = np.loadtxt('../data/Post_Russia_Recordings_csv/csvs/' + str(i) + '.csv', delimiter=',', dtype=np.float32)
#     result = np.concatenate([result, np.zeros(shape=(l - result.shape[0], 80), dtype=np.float32)], axis=0)
#     tensor.append(result.T.reshape(80, -1))
# tensor = np.array(tensor)
#
# # tensor = np.loadtxt('1wav.csv', delimiter=',', dtype=np.float32).T.reshape(1, 80, -1)
# Nw = Network()
# res = Nw.apply(tensor)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#