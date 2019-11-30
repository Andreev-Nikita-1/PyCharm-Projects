import numpy as np
import onnx
import onnxruntime

letters = []
fi = open("letters.lst", 'r', encoding='utf8')
for line in fi.readlines():
    letters.append(line[0])
letters = np.array(letters)


class Network:
    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession("quartznet.onnx")
        for nn_input in self.onnx_session.get_inputs():
            print(nn_input)

    def apply(self, features):
        ort_outs = np.array(self.onnx_session.run(None, {"features": features}))
        return ort_outs.reshape(features.shape[0], -1, 36)


l = 896
tensor = []
for i in range(51, 71):
    result = np.loadtxt('../data/Post_Russia_Recordings_csv/csvs/' + str(i) + '.csv', delimiter=',', dtype=np.float32)
    result = np.concatenate([result, np.zeros(shape=(l - result.shape[0], 80), dtype=np.float32)], axis=0)
    tensor.append(result.T.reshape(80, -1))
tensor = np.array(tensor)

# tensor = np.loadtxt('1wav.csv', delimiter=',', dtype=np.float32).T.reshape(1, 80, -1)
Nw = Network()
res = Nw.apply(tensor)

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
