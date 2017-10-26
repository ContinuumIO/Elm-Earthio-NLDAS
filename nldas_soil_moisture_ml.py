from __future__ import print_function, unicode_literals, division

from collections import OrderedDict
import datetime
from functools import partial
from itertools import product
import os

import dill
from elm.pipeline import Pipeline
from elm.pipeline.steps import (linear_model,
                                decomposition,
                                gaussian_process,
                                preprocessing)
from elm.pipeline.predict_many import predict_many
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from elm.model_selection.sorting import pareto_front
from elm.model_selection import EaSearchCV
import numpy as np
from xarray_filters import MLDataset
from xarray_filters.pipeline import Generic, Step

from read_nldas_forcing import (slice_nldas_forcing_a,
                                GetY, FEATURE_LAYERS,
                                SOIL_MOISTURE)
from nldas_soil_features import nldas_soil_features
from ts_raster_steps import differencing_integrating
from changing_structure import ChooseWithPreproc

NGEN = 3
NSTEPS = 1
WATER_MASK = -9999
DEFAULT_CV = 5
X_TIME_STEPS = 144

BASE_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/NLDAS/{}/{:04d}/{:03d}/{}'
START_DATE = datetime.datetime(2000, 1, 1, 1, 0, 0)

DEFAULT_MAX_STEPS = 144
ONE_HR = datetime.timedelta(hours=1)
TIME_OPERATIONS = ('mean',
                   'std',
                   'sum',
                   ('diff', 'mean'),
                   ('diff', 'std'),
                   ('diff', 'sum'))
REDUCERS = [('mean', x) for x in TIME_OPERATIONS if x != 'mean']

np.random.seed(42)  # TODO remove

def log_trans_only_positive(X, y, **kw):
    Xnew = OrderedDict()
    for j in range(X.features.shape[1]):
        minn = X.features[:, j].min().values
        if minn <= 0:
            continue
        X.features.values[:, j] = np.log10(X.features.values[:, j])
    return X, y


class Flatten(Step):
    def transform(self, X, y=None, **kw):
        return X.to_features(), y


class DropNaRows(Step):
    def transform(self, X, y=None, **kw):
        return X, y


class Differencing(Step):
    hours_back = 144
    first_bin_width = 12
    last_bin_width = 1
    num_bins = 12
    bin_shrink = 'linear'
    reducers = 'mean'
    layers = None

    def transform(self, X, y=None, **kw):
        return differencing_integrating(X, **self.get_params())


SOIL_PHYS_CHEM = {}
class AddSoilPhysicalChemical(Step):
    add = True
    soils_dset = None
    to_raster = True
    avg_cos_hyd_params = True
    kw = None
    def transform(self, X, y, **kw):
        global SOIL_PHYS_CHEM
        params['kw'] = params['kw'] or {}
        params = self.get_params().copy()
        if not params.pop('add'):
            return X, y
        hsh = hash(repr(params))
        if hsh in SOIL_PHYS_CHEM:
            soils = SOIL_PHYS_CHEM[hsh]
        else:
            soils = soil_features(**params)
            if len(SOIL_PHYS_CHEM) < 3:
                SOIL_PHYS_CHEM[hsh] = soils
        return MLDataset(xr.merge(soils, X))

SCALERS = [preprocessing.StandardScaler()] + [preprocessing.MinMaxScaler()] * 10

param_distributions = {
    'scaler___estimator': SCALERS,
    'scaler___trans': [log_trans_only_positive],
    'pca__n_components': [6, 7, 8, 10, 14, 18],
    'pca__estimator': [decomposition.PCA(),
                      decomposition.FastICA(),
                      decomposition.KernelPCA()],
    'pca__run': [True, True, False],
    'time__hours_back': [int(DEFAULT_MAX_STEPS / 2), DEFAULT_MAX_STEPS],
    'time__last_bin_width': [1,],
    'time__num_bins': [4,],
    'time__weight_type': ['uniform', 'log', 'log', 'linear', 'linear'],
    'time__bin_shrink': ['linear', 'log'],
    'time__reducers': REDUCERS,
    'soil_phys__add': [True, True, True, False],
}

model_selection = {
    'select_method': 'selNSGA2',
    'crossover_method': 'cxTwoPoint',
    'mutate_method': 'mutUniformInt',
    'init_pop': 'random',
    'indpb': 0.5,
    'mutpb': 0.9,
    'cxpb':  0.3,
    'eta':   20,
    'ngen':  2,
    'mu':    16,
    'k':     8, # TODO ensure that k is not ignored - make elm issue if it is
    'early_stop': None
}

def get_file_name(tag, date):
    date = date.isoformat().replace(':','_').replace('-','_')
    return '{}-{}.dill'.format(tag, date)


def dump(obj, tag, date):
    fname = get_file_name(tag, date)
    return getattr(obj, 'dump', getattr(obj, 'to_netcdf'))(fname)


def main(date=START_DATE, cv=DEFAULT_CV):
    '''
    Beginning on START_DATE, step forward hourly, training on last
    hour's NLDAS FORA dataset with transformers in a 2-layer hierarchical
    ensemble, training on the last hour of data and making
    out-of-training-sample predictions for the current hour.  Makes
    a dill dump file for each hour run. Runs fro NSTEPS hour steps.
    '''
    estimators = []
    for step in range(NSTEPS):
        out = train_one_time_step(date,
                                  cv=DEFAULT_CV,
                                  estimators=estimators)
        ea, X, second_layer, pred, pred_layer_2, pred_avg = out
        scores = pd.DataFrame(ea.cv_results_)
        scores.to_pickle(get_file_name('scores', date))
        pred.to_netcdf(get_file_name('pred_layer_1', date))
        pred_layer_2 = second_layer.predict(X)
        pred_layer_2.to_netcdf(get_file_name('pred_layer_2', date))
        pred_avg = (pred + pred_layer_2) / 2.
        pred_avg.to_netcdf(get_file_name('pred_avg', date))
    return ea, X, second_layer, pred, pred_layer_2, pred_avg

class Sampler(Step):
    date = None
    def transform(self, X, y=None, **kw):
        if X is None:
            X = self.date
        return slice_nldas_forcing_a(X, X_time_steps=max_time_steps)


def new_ea_search_pipeline(splits, y_layer=SOIL_MOISTURE, cv=DEFAULT_CV):
    #X, y = pipe.fit()
    pipe = Pipeline([
        ('time', Differencing(layers=FEATURE_LAYERS)),
        ('flatten', Flatten()),
        ('soil_phys', AddSoilPhysicalChemical()),
        ('drop_null', DropNaRows()),
        ('get_y', GetY(y_layer)),
        ('scaler', ChooseWithPreproc(trans_if=log_trans_only_positive)),
        ('pca', ChooseWithPreproc()),
        ('estimator', linear_model.LinearRegression(n_jobs=-1))])

    ea = EaSearchCV(pipe,
                    param_distributions=param_distributions,
                    sampler=Sampler(),
                    ngen=NGEN,
                    model_selection=model_selection,
                    cv=cv)
    return ea

from sklearn.model_selection import KFold

max_time_steps = DEFAULT_MAX_STEPS // 2
date = START_DATE
dates = np.array([START_DATE - datetime.timedelta(hours=hr)
                 for hr in range(max_time_steps)])
splits = list(KFold(DEFAULT_CV).split(dates))

#dummy_X = np.random.uniform(0, 1, (1000, 2))
#dummy_y = np.random.uniform(0, 1, (1000,))


ea = new_ea_search_pipeline(splits)
print(ea.get_params())
date += ONE_HR
#print(ea.get_params())
current_file = get_file_name('fit_model', date)
ea.fit(dates)
dump(ea, tag, date)
estimators.append(ea)
second_layer = MultiLayer(estimator=linear_model.LinearRegression,
                          estimators=estimators)
second_layer.fit(X)
pred = ea.predict(X)
  #  return (ea, X, second_layer,
   #         pred, pred_layer_2, pred_avg,
    #        date, current_file)


#if __name__ == '__main__':
    #ea, X, second_layer, pred, pred_layer_2, pred_avg = main()

