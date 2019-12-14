import scipy.sparse
import sklearn.datasets
import argparse
import json

from .oracle import *
from .optimization_method import *

parser = argparse.ArgumentParser()
parser.add_argument("--ds_path", help="path to dataset file in .svm format", type=str)
parser.add_argument("--optimize_method",
                    help="high-level optimization method, will be one of {'gradient', 'newton', 'hfn', 'BFGS', 'L-BFGS', 'lasso'}",
                    type=str)
parser.add_argument("--line_search",
                    help="linear optimization method, will be one of {'golden_search', 'brent', 'armijo', 'wolfe', 'lipschitz'}",
                    type=str)
parser.add_argument("--point_distribution",
                    help="initial weights distribution class, will be one of {'uniform', 'gaussian'}", type=str)
parser.add_argument("--seed", help="seed for numpy randomness", type=int)
parser.add_argument("--eps", help="epsilon to use in termination condition", type=float)
parser.add_argument("--cg_tolerance_policy",
                    help="optional key for HFN method; conjugate gradients method tolerance choice policy, will be one of {'const', 'sqrtGradNorm', 'gradNorm'}",
                    type=str)
parser.add_argument("--cg_tolerance_eta",
                    help="optional key for HFN method; conjugate gradients method tolerance parameter eta", type=float)
parser.add_argument("--lbfgs_history_size",
                    help="optional key for L-BFGS method; history size", type=int)
parser.add_argument("--lasso_coeff",
                    help="optional key for lasso method; coefficient l", type=float)

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

    method = args.optimize_method
    search_kwargs = {}
    if method == 'lasso':
        if args.lasso_coeff is not None:
            search_kwargs = {'l': args.lasso_coeff}

    linear_solver = 'cholesky' if args.optimize_method == 'newton' else 'cg'
    solver_kwargs = dict(
        [('eta', args.cg_tolerance_eta), ('policy', args.cg_tolerance_policy)]) if linear_solver == 'cg' else dict([])

    if args.point_distribution == 'uniform':
        w0 = (2 * np.random.random(X.shape[1]) - 1) / 2
    else:
        w0 = np.random.normal(0, 1, X.shape[1])

    oracle = Oracle(X, labels)

    sol = optimization_task(oracle, w0, method=method, linear_solver=linear_solver,
                            one_dim_search=args.line_search,
                            epsilon=args.eps,
                            search_kwargs=search_kwargs,
                            solver_kwargs=solver_kwargs
                            )

    answer = json.dumps(
        {'initial_point': str(w0), 'optimal_point': str(sol['x']), 'func_value': sol['fun'],
         'gradient_value': str(sol['jac']),
         'oracle_calls': {'f': sol['nfev'], 'df': sol['njev'], 'd2f': sol['nhev']}, 'r_k': sol['ratio'],
         'working_time': sol['time']}, indent=4)

    print(answer)


main()
