import numpy as np
import tensorflow.compat.v1 as tf

data = np.loadtxt('1wav.csv', delimiter=',')


class Loader:
    def __init__(self, path):
        self.load_graph(path)

    def load_graph(self, model_filepath):
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=False))

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        self.input_features = tf.placeholder(tf.float16, shape=[1, 80, 514], name='input_features')
        self.input_lengths = tf.placeholder(tf.int32, shape=[1, 1], name='input_lengths')

        tf.import_graph_def(graph_def, {'input_features': self.input_features, 'input_lengths': self.input_lengths})

        print('Model loading complete!')

    def test(self, data):
        output_tensor = self.graph.get_tensor_by_name("import/cnn/output:0")
        output = self.sess.run(output_tensor,
                               feed_dict={self.input_features: data, self.input_lengths: np.array([[514]])})

        return output

loader = Loader('saved_model.pb')

exit(0)
