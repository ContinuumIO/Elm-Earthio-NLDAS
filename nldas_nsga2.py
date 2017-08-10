from __future__ import print_function, unicode_literals, division
from dask.distributed import Client
from elm.model_selection.evolve import ea_setup
from elm.pipeline.evolve_train import evolve_train
from sklearn.neural_network import MLPRegressor

from nldas_soil_moisture_ml import *

def subset_ensemble_init_func(*args, **kw):
    ensemble = ensemble_init_func(*args, **kw)
    np.random.shuffle(ensemble)
    return ensemble

client = None#Client()

quantiles = [(0.25, 0.75)] * 5
quantiles += [(0.2, 0.8)] * 3
with_cen = [True, False] * len(quantiles)
with_cen = with_cen[:len(quantiles)]
robust_scalers = [('robust', steps.RobustScaler(quantile_range=(mn, mx), with_centering=w))
                  for mn, mx in quantiles
                  for w in with_cen]

nsga_control = {
    'select_method': 'selNSGA2',
    'crossover_method': 'cxTwoPoint',  # TODO can we modify for float/int
    'mutate_method': 'mutUniformInt',  # TODO same comment as ^^
    'init_pop': 'random',
    'indpb': 0.6,     # probability of each attribute changing
    'mutpb': 0.9,     # probability of mutation
    'cxpb':  0.5,     # probability of crossover
    'eta':   20,      # eta: Crowding degree of the crossover
                      # A high eta will produce children resembling
                      # to their parents, while a small eta will
                      # produce solutions much more different.
    'ngen':  3,       # Number of generations
    'mu':    64,      # Population size
    'k':     24,      # Number selected to move to next generation
    'early_stop': {'threshold': [0, 1, 1], 'agg': all},
    # alternatively 'early_stop': {'abs_change': [10], 'agg': 'all'},
    # alternatively early_stop: {percent_change: [10], agg: all}
    # alternatively early_stop: {threshold: [10], agg: any}
}
param_grid =  {
    'diff_avg__X_time_averaging': avg_scenarios,
    'MinMaxScaler__feature_range': minmax_bounds,
    'pca__n_components': [4, 5, None],#n_components,
    'estimator__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1,
                         1, 10, 100, 1000, 10000, 1000000],
    #'weights__no_weights': [False],
    'control': nsga_control,
}

evo_params = ea_setup(param_grid=param_grid,
                      param_grid_name='nldas_param_grid_example',
                      score_weights=[-1, 1, 1]) # MSE, R2, True/False is ok

init_func = partial(ensemble_init_func,
                    pca=pca,
                    scalers=robust_scalers,
                    n_components=n_components,
                    estimators=estimators,
                    preamble=preamble,
                    log=log,
                    diff_avg_hyper_params=diff_avg_hyper_params,
                    weights_kw=[dict(no_weights=True), dict()],
                    summary='TODO fix')

diff_kw = diff_avg_hyper_params[0]

last_hour_data = sampler(START_DATE - ONE_HR)
this_hour_data = sampler(START_DATE)

data_source = dict(X=last_hour_data)
example_raster = last_hour_data.data_vars[tuple(last_hour_data.data_vars.keys())[0]]
not_null_count = example_raster.values[~np.isnan(example_raster)].size
batch_size = int(0.7 * not_null_count)
mlp_params = dict(batch_size=batch_size, warm_start=True, alpha=1)
pipe = Pipeline(preamble(diff_kw) +
                [log, robust_scalers[0], pca(),
                ('estimator', MLPRegressor(**mlp_params))],
                **pipeline_kw)

nsga2_func = partial(evolve_train, pipe, evo_params,
                      client=client,
                      saved_ensemble_size=nsga_control['mu'],
                      ensemble_init_func=init_func,
                      method='fit',
                      **data_source)
out = train_model_on_models(last_hour_data, this_hour_data,
                            init_func, nsga2_func=nsga2_func)
last_hour_data, this_hour_data, models, preds, models2, preds2 = out

