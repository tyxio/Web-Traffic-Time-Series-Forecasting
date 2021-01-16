from mypackage.utils import *
from mypackage.sarima import *
from mypackage.naive import *
from mypackage.mylogging import *
from mypackage.settings import initialize_settings, smapes
import mypackage.config as config


import os
import ast
import time
import json
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings

# score a model, return None on failure
run = 1

def run_model(train, test, predictions, colname, cfg, mode='validation'):
    order, sorder, trend = cfg

    try:
        if (trend == 'naive1'):
            result = use_last_value(train, test, predictions, colname, mode)
        elif (trend == 'naive2'):
            result = moving_average(train, test, predictions, colname, mode)
        elif (trend == 'naive3'):
            result = simple_exp_smoothing(train, test, predictions, colname, mode)
        elif (trend == 'naive4'):
            result = holt_linear_trend(train, test, predictions, colname, mode)
        elif (trend == 'naive5'):
            result = holt_exp_smoothing(train, test, predictions, colname, mode)
        else:
            # SARIMA grid search
            result = walk_forward_validation(
                    train[colname], test[colname], predictions[colname], cfg, mode)
    except:
        result = None

    return result

def score_model(train, test, predictions, colname, cfg, mode='validation'):
    global run
    result = None
    key = str(cfg)

    result = run_model(train, test, predictions, colname, cfg, mode)
    
    # check for an interesting result
    if result is not None and result != 200:
        print(' > [run %d] [%s] Model[%s] %.3f' % (run, colname, key, result))
    
    run += 1

    return (key, result)

# grid search


def grid_search(train, test, predictions, colname, cfg_list, mode='validation', parallel=True):
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(train, test, predictions,
                                      colname, cfg, mode) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(train, test, predictions, colname,
                              cfg, mode=mode) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def train_one_series(train, test, predictions, colname, cfg_list, mode, plot=False):
    global smapes

    # find the best sarima model
    scores = grid_search(train, test, predictions, colname,
                         cfg_list, mode=mode, parallel=False)

    # top config
    cfg, error = scores[0]
    smapes = np.append(smapes, [error])

    result_folder = os.path.join(FIGURES_FOLDER, cfg)       
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # make a final prediction with the top configuration and plot
    if (plot == True):
        if ('naive' not in cfg):
            best_config = list()
            best_config.append(ast.literal_eval(cfg))
            grid_search(train, test, predictions, colname,
                        best_config, mode=mode, parallel=False)       
        
        plot_estimate2(train, test, predictions, colname, title='Forecasts', config=cfg)

    # return the best sarima configuration
    return cfg


def train_all_series(cfg_list):
    global run
    mode = 'validation'

    print('Prepare the train/test/predictions frames (mode:%s)...' % (mode))
    train, test, predictions = prepare_data(
        data, remove_outliners=config.REMOVE_OUTLINERS, mode=mode)

    all_best_configs = list()

    for (colname, _) in train.iteritems():
        run = 0
        # train?
        if len(cols_to_model) > 0 and colname not in cols_to_model:
            continue
        # plot?
        plot = False
        if colname in cols_to_plot:
            plot = True

        # train this series
        best_config = train_one_series(
            train, test, predictions, colname, cfg_list, mode=mode, plot=plot)
        all_best_configs.append(best_config)

    result_folder = os.path.join(FIGURES_FOLDER, 'best')       
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    report_results(train, test, predictions, smapes,
                   plot_smapes=True, plot_good_predictions=True, plot_bad_predictions=True, config='best')

    return all_best_configs


def calculate_kaggle_contest(all_best_configs):
    mode = 'kaggle'

    train, test, predictions = prepare_data(
        data, remove_outliners=config.REMOVE_OUTLINERS, mode=mode)

    n = 0
    for (colname, _) in train.iteritems():
        if len(cols_to_model) > 0 and colname not in cols_to_model:
            continue
        print(colname)
   
        arr = np.array(ast.literal_eval(all_best_configs[n]), dtype=list)
        sarima_params = np.array_split(arr, 3)
        sarima_params[0] = sarima_params[0][0]
        sarima_params[1] = sarima_params[1][0]
        sarima_params[2] = sarima_params[2][0]
        run_model(train, test, predictions, colname, sarima_params, mode)
        plot_estimate2(train, test, predictions, colname, title='Forecasts', config=all_best_configs[0])
        n += 1

    print(predictions.shape)
    kaggle_format_and_dump(
        predictions, config.KAGGLE_FILE)


if __name__ == '__main__':

    # ************************
    # PREPARE
    # ************************
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    print('Change dir to current folder ' + os.getcwd())

    print('Import the web traffic time series...')
    data = pd.read_csv(config.DATA_FILE,
                       index_col="Day", parse_dates=True)
    data.shape

    print('Initialize training')
    initialize_settings() 

    cfg_list = sarima_configs(test=config.USE_TEST_CONFIG)
    print('cfg_list len:', len(cfg_list))

    # ************************
    # GRID SEARCH
    # ************************
    sarima_params = list()
    if (config.RUN_GRID_SEARCH):
        print('Launch grid search')
        start = time.time()
        sarima_params = train_all_series(cfg_list)
        end = time.time()
        print(f'Grid search done in {end-start} secs')

        # save list of parameters to file
        with open(config.PARAMS_FILE, 'w') as filehandle:
            json.dump(sarima_params, filehandle)

    # ************************
    # KAGGLE CONTEST
    # ************************
    if (config.RUN_KAGGLE_CONTEST):
        print('calculate_kaggle_contest')

        with open(config.PARAMS_FILE, 'r') as filehandle:
            sarima_params = json.load(filehandle)

        if (len(sarima_params) == 0):
            sarima_params = [
                config.SARIMA_DEFAULT_PARAMETERS for i in range(data.shape[0])]

        calculate_kaggle_contest(sarima_params)

    plt.show()
