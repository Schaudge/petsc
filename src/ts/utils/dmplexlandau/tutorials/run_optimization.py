from objective import *
import time

from skopt.space import Integer
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

def main():
    tic = time.perf_counter()

    team_size_min = 1
    team_size_max = 1.1
    vector_length_min = 1
    vector_length_max = 1.1
    nsolves_team_min = 1
    nsolves_team_max = 1.1
    max_size = 8
    extra_options = ''
    n_calls = 32
    n_random_starts = 1


    dim1 = Integer(team_size_min, team_size_max, name='team_size')
    dim2 = Integer(vector_length_min, vector_length_max, name='vector_length')
    dim3 = Integer(nsolves_team_min, nsolves_team_max, name='nsolves_team')
    dim4 = Integer(0, 1, name='ortho_strategy')
    dim5 = Integer(0, 1, name='temp_data_strategy')
    dim6 = Integer(0, 1, name='shared_level')

    dimensions = [dim1, dim2, dim3, dim4, dim5, dim6]

    # define fixed params
    fixed_args = FixedParams(max_size, {}, 'log.txt', extra_options)

    @use_named_args(dimensions)
    def objective(team_size, vector_length, nsolves_team, ortho_strategy, temp_data_strategy, shared_level):
        variable_args = VariableParams(team_size, vector_length, nsolves_team, ortho_strategy, temp_data_strategy, shared_level)
        return objective_namedtuple(variable_args, fixed_args)

    kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10,
                                normalize_y=True, noise=None,
                                n_restarts_optimizer=2
                                )

    res_gp = gp_minimize(objective, dimensions, base_estimator=gpr,
        n_calls=n_calls, n_random_starts=n_random_starts, random_state=0)
    toc = time.perf_counter()

    print(res_gp)

    with open('results.txt', 'w') as f:
        print(res_gp, file=f)
        print("Elapsed time = " + str(toc-tic) + " seconds", file=f)
        print(res_gp.x, file=f)
        print(res_gp.fun, file=f)

if __name__ == "__main__":
    main()
