import scipy.sparse
import sklearn.datasets
import argparse
import json
import warnings

# warnings.filterwarnings("ignore")

from .oracle import *
from .optimization_method import *

parser = argparse.ArgumentParser()
parser.add_argument("--ds_path", help="path to dataset file in .svm format", type=str)
parser.add_argument("--optimize_method",
                    help="high-level optimization method, will be one of {'gradient', 'newton', 'hfn', 'BFGS', 'L-BFGS', 'l1prox', 'sgd'}",
                    type=str, default='hfn')
parser.add_argument("--line_search",
                    help="linear optimization method, will be one of {'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}",
                    type=str)
parser.add_argument("--point_distribution",
                    help="initial weights distribution class, will be one of {'uniform', 'gaussian'}", type=str)
parser.add_argument("--seed", help="seed for numpy randomness", type=int)
parser.add_argument("--eps", help="epsilon to use in termination condition", type=float, default=1e-4)
parser.add_argument("--cg_tolerance_policy",
                    help="optional key for HFN method; conjugate gradients method tolerance choice policy, will be one of {'const', 'sqrtGradNorm', 'gradNorm'}",
                    type=str, default='sqrtGradNorm')
parser.add_argument("--cg_tolerance_eta",
                    help="optional key for HFN method; conjugate gradients method tolerance parameter eta", type=float,
                    default=0.5)
parser.add_argument("--lbfgs_history_size",
                    help="optional key for L-BFGS method; history size", type=int, default=20)
parser.add_argument("--l1_lambda", help="optional key for l1-proximal method; l1-regularization coefficient.",
                    type=float, default=0)
parser.add_argument("--batch_size", help="optional key for SGD method.",
                    type=int, default=20)
parser.add_argument("--SGD_iters", help="number of SGD method iterations",
                    type=int, default=1000)

args = parser.parse_args()


def load_data(path):
    data = sklearn.datasets.load_svmlight_file(path)
    X = data[0]
    dummy = scipy.sparse.csr_matrix([[1] for i in range(X.shape[0])])
    X = scipy.sparse.hstack([X, dummy])
    labels = data[1]
    d = dict([(l, 2 * y - 1) for y, l in enumerate(np.unique(labels))])
    labels = np.array([d[l] for l in labels])
    return X, labels


def main():
    np.random.seed(args.seed)
    X, labels = load_data(args.ds_path)

    if args.line_search == 'golden_search' or args.line_search == 'brent':
        print('not supported')
        exit(0)

    method = args.optimize_method
    search_kwargs = {}
    solver_kwargs = dict(
        [('eta', args.cg_tolerance_eta), ('policy', args.cg_tolerance_policy)])
    oracle = Oracle(scipy.sparse.csr_matrix(X), labels)

    if method == 'sgd':
        sol = SGD(oracle, batch_size=args.batch_size, max_iter=args.SGD_iters)
    else:
        sol = optimization_task(oracle, method=method,
                                one_dim_search=args.line_search,
                                epsilon=args.eps,
                                search_kwargs=search_kwargs,
                                solver_kwargs=solver_kwargs,
                                m=args.lbfgs_history_size,
                                l=args.l1_lambda
                                )

    answer = json.dumps(
        {'initial_point': str(sol['start']), 'optimal_point': str(sol['x']), 'func_value': sol['fun'],
         'gradient_value': str(sol['jac']),
         'oracle_calls': {'f': sol['nfev'], 'df': sol['njev'], 'd2f': sol['nhev']}, 'r_k': sol['ratio'],
         'working_time': sol['time'], 'null_compontns': str(sol['null_components'])}, indent=4)

    print(answer)


main()

