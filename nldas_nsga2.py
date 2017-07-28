from dask.distributed import Client
from elm.model_selection.evolve import ea_setup
from elm.pipeline.evolve_train import evolve_train

from nldas_soil_moisture_ml import *

def subset_ensemble_init_func(*args, **kw):
    ensemble = ensemble_init_func(*args, **kw)
    np.random.shuffle(ensemble)
    return ensemble

client = None#Client()

data_source = dict(sampler=sampler, args_list=[START_DATE])

nsga_control = {
    'select_method': 'selNSGA2',
    'crossover_method': 'cxTwoPoint',
    'mutate_method': 'mutUniformInt',
    'init_pop': 'random',
    'indpb': 0.5,     # probability of each attribute changing
    'mutpb': 0.9,     # probability of mutation
    'cxpb':  0.3,     # probability of crossover
    'eta':   20,      # eta: Crowding degree of the crossover
                      # A high eta will produce children resembling
                      # to their parents, while a small eta will
                      # produce solutions much more different.
    'ngen':  6,       # Number of generations
    'mu':    24,      # Population size
    'k':     8,       # Number selected to move to next generation
    'early_stop': {'threshold': [0, 1, 1], 'agg': all},
    # alternatively 'early_stop': {'abs_change': [10], 'agg': 'all'},
    # alternatively early_stop: {percent_change: [10], agg: all}
    # alternatively early_stop: {threshold: [10], agg: any}
}
param_grid =  {
    'diff_avg__X_time_averaging': avg_scenarios,
    'MinMaxScaler__feature_range': minmax_bounds,
    'pca__n_components': n_components,
    'weights__no_weights': [True, False],
    'control': nsga_control,
}

evo_params = ea_setup(param_grid=param_grid,
                      param_grid_name='nldas_param_grid_example',
                      score_weights=[-1, 1, 1])

init_func = partial(ensemble_init_func,
                    pca=pca,
                    scalers=minmax_scalers,
                    n_components=n_components,
                    estimators=estimators,
                    preamble=preamble,
                    log=log,
                    diff_avg_hyper_params=diff_avg_hyper_params,
                    weights_kw=[dict(no_weights=True), dict()],
                    summary='TODO fix')

diff_kw = diff_avg_hyper_params[0]
pipe = Pipeline(preamble(diff_kw, dict(no_weights=True)) +
                [log, minmax_scalers[0], pca(),
                ('estimator', LinearRegression(n_jobs=-1))],
                **pipeline_kw)

models = evolve_train(pipe, evo_params,
                      client=client,
                      saved_ensemble_size=nsga_control['mu'],
                      ensemble_init_func=init_func,
                      method='fit',
                      partial_fit_batches=1,
                      **data_source)