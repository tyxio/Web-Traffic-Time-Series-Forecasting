RUN_GRID_SEARCH=True
RUN_KAGGLE_CONTEST=True

REMOVE_OUTLINERS=False
USE_TEST_CONFIG=True
TRANSFORM_DATA=False
SARIMA_DEFAULT_PARAMETERS="[(2,0,1), (0,1,2,7), 't']"

# specify which series to train and plot
#cols_to_model = train.columns
#cols_to_plot = []
#cols_to_model = ['series-10', 'series-19', 'series-20', 'series-40', 'series-44']
#cols_to_plot = ['series-10', 'series-19', 'series-20', 'series-40', 'series-44']
cols_to_model = ['series-50', 'series-20']
cols_to_plot = ['series-50', 'series-20']

DATA_FILE=r'data/train.csv'
RESULTS_FOLDER=r'results/series-20/'
KAGGLE_FILE= RESULTS_FOLDER + r'kaggle_forecasts.csv'
PARAMS_FILE=RESULTS_FOLDER + r'params.json'
FIGURES_FOLDER=RESULTS_FOLDER
SMAPES_FOLDER=RESULTS_FOLDER