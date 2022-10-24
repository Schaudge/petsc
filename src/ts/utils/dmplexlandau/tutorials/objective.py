from run_ex2 import run_test
from collections import namedtuple


def objective_namedtuple(params, fixed_params):
    if params.team_size*params.vector_length > fixed_params.max_size:
        return 0

    key = str(params.team_size)+'_'+\
          str(params.vector_length)+'_'+\
          str(params.nsolves_team)+'_'+\
          str(params.ortho_strategy)+'_'+\
          str(params.temp_data_strategy)+'_'+\
          str(params.shared_level)

    if key in fixed_params.previous_points:
        print('point '+key+ ' was already evaluated.')
    else:
        try:
            data = run_test(team_size=params.team_size, vector_length=params.vector_length,\
                            nsolves_team=params.nsolves_team, ortho_strategy=params.ortho_strategy,\
                            temp_data_strategy=params.temp_data_strategy, shared_level=params.shared_level,\
                            extra_options=fixed_params.extra_options)
            fixed_params.previous_points[key] = -1/data
        except:
            fixed_params.previous_points[key] = 0.


    with open(fixed_params.log_file_name, 'w') as f:
        for key, value in fixed_params.previous_points.items():
            print(key, ' : ', value, file=f)    

    return fixed_params.previous_points[key]

VariableParams = namedtuple('VariableParams', 'team_size vector_length nsolves_team ortho_strategy temp_data_strategy shared_level')
FixedParams = namedtuple('FixedParams', 'max_size previous_points log_file_name extra_options')
