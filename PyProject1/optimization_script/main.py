import scipy.sparse
import sklearn.datasets
import argparse
from optimization_script.oracle import *
from optimization_script.optimization_method import *

parser = argparse.ArgumentParser()
parser.add_argument("--ds_path", help="path to dataset file in .svm format", type=str)
parser.add_argument("--optimize_method",
                    help="high-level optimization method, will be one of {'gradient', 'newton', 'hfn'}", type=str)
parser.add_argument("--line_search",
                    help="linear optimization method, will be one of {'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}",
                    type=str)
parser.add_argument("--point_distribution",
                    help="initial weights distribution class, will be one of {'uniform', 'gaussian'}", type=str)
parser.add_argument("--seed", help="seed for numpy randomness", type=int)
parser.add_argument("--eps", help="epsilon to use in termination condition", type=float)
parser.add_argument("--cg-tolerance-policy",
                    help="optional key for HFN method; conjugate gradients method tolerance choice policy, will be one of {'const', 'sqrtGradNorm', 'gradNorm'}",
                    type=str)
parser.add_argument("--cg-tolerance-eta",
                    help="optional key for HFN method; conjugate gradients method tolerance parameter eta", type=float)
args = parser.parse_args()
np.random.seed(0)
print(np.random.random())
print(np.random.random())
print(np.random.random())
print(np.random.random())
randd()
randd()
randd()
randd()
randd()


def main():
    method = 'gradient descent'
    one_dim_search = 'wolf'
    seed = 0
    epsilon = 0.00001
    distr = 'uniform'

    for arg in sys.argv[1:]:
        name, value = arg.split('=')
        if name == '--ds_path':
            path = value
        elif name == '--optimize_method':
            if value == 'gradient':
                method = 'gradient descent'
            else:
                method = 'newton'
        elif name == '--line_search':
            if value == 'golden_search':
                one_dim_search = 'golden'
            elif value == 'brent':
                one_dim_search = 'brent'
            elif value == 'armijo':
                one_dim_search = 'armiho'
            elif value == 'wolfe':
                one_dim_search = 'wolf'
            elif value == 'lipschitz':
                one_dim_search = 'nester'
        elif name == '--seed':
            seed = value
        elif name == '--eps':
            epsilon = float(value)
        elif name == '--point_distribution':
            if value == 'uniform':
                distr = 'uniform'
            else:
                distr = 'normal'

    data = sklearn.datasets.load_svmlight_file(path)
    X = data[0]
    dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
    X = scipy.sparse.hstack([X, dummy])
    labels = data[1]
    d = dict([(l, 2 * y - 1) for y, l in enumerate(np.unique(labels))])
    labels = np.array([d[l] for l in labels])

    np.random.seed(int(seed))
    if distr == 'uniform':
        w0 = (2 * np.random.random(X.shape[1]) - 1) / 2
    else:
        w0 = np.random.normal(0, 0.5, X.shape[1])

    outers = scipy.sparse.csr_matrix([np.outer(x, x).flatten() for x in X.todense()])

    w, fw, k, r, t, fc, gc, hc = optimization_task(oracle, w0, method=method, one_dim_search=one_dim_search,
                                                   args=[X, labels, outers], epsilon=epsilon)
    answer = '{\n' + \
             '\t \'initial_point\': ' + '\'' + str(w0) + '\',\n' + \
             '\t \'optimal_point\': ' + '\'' + str(w) + '\',\n' + \
             '\t \'function_value\': ' + '\'' + str(fw) + '\',\n' + \
             '\t \'function_calls\': ' + '\'' + str(fc) + '\',\n' + \
             '\t \'gradient_calls\': ' + '\'' + str(gc) + '\',\n' + \
             '\t \'hessian_calls\': ' + '\'' + str(hc) + '\',\n' + \
             '\t \'r_k\': ' + '\'' + str(r) + '\',\n' + \
             '\t \'working_time\': ' + '\'' + str(t) + '\'\n' + \
             '}'

    print(answer)


print('hi')
# if __name__ == '__main__':
#     main()
