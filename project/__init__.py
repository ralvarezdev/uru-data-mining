import os
import re

# Constants
CWD= os.getcwd()
DATASET_DIR = os.path.join(CWD,'dataset')
ROAD_ACCIDENTS_DIR = os.path.join(DATASET_DIR, 'road-accidents')
ROAD_ACCIDENT_DATASET_FILE = os.path.join(ROAD_ACCIDENTS_DIR,'road_accident_dataset.csv')
TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE = os.path.join(ROAD_ACCIDENTS_DIR,'transformed_road_accident_dataset.csv')
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTOGRAMS_DIR = os.path.join(PROJECT_DIR, 'histograms')
BOXPLOTS_DIR=os.path.join(PROJECT_DIR, 'boxplots')
COUNTPLOTS_DIR=os.path.join(PROJECT_DIR, 'countplots')
MUTUAL_REGRESSIONS_DIR=os.path.join(PROJECT_DIR, 'mutual_regressions')
OLS_DIR= os.path.join(PROJECT_DIR, 'ols')
PAIRPLOTS_DIR=os.path.join(PROJECT_DIR, 'pairplots')
HEATMAPS_DIR=os.path.join(PROJECT_DIR, 'heatmaps')
RIDGE_REGRESSIONS_DIR=os.path.join(PROJECT_DIR, 'ridge_regressions')
RIDGE_REGRESSIONS_FILE=os.path.join(RIDGE_REGRESSIONS_DIR,'ridge_regressions.txt')
LASSO_REGRESSIONS_DIR=os.path.join(PROJECT_DIR, 'lasso_regressions')
LASSO_REGRESSIONS_FILE=os.path.join(LASSO_REGRESSIONS_DIR,'lasso_regressions.txt')
ELASTIC_NET_REGRESSIONS_DIR=os.path.join(PROJECT_DIR, 'elastic_net_regressions')
ELASTIC_NET_REGRESSIONS_FILE=os.path.join(ELASTIC_NET_REGRESSIONS_DIR,'elastic_net_regressions.txt')
RANDOM_FOREST_DIR=os.path.join(PROJECT_DIR, 'random_forest')
RANDOM_FOREST_FILE=os.path.join(RANDOM_FOREST_DIR,'random_forest.txt')
PREVIEW_ROWS = 10
COLUMNS_CHUNK_SIZE = 5
COLUMNS = ['Country', 'Year', 'Month', 'Day of Week', 'Time of Day', 'Urban/Rural', 'Road Type', 'Weather Conditions', 'Visibility Level', 'Number of Vehicles Involved', 'Speed Limit', 'Driver Age Group', 'Driver Gender', 'Driver Alcohol Level', 'Driver Fatigue', 'Vehicle Condition', 'Pedestrians Involved', 'Cyclists Involved', 'Accident Severity', 'Number of Injuries', 'Number of Fatalities', 'Emergency Response Time', 'Traffic Volume', 'Road Condition', 'Accident Cause', 'Insurance Claims', 'Medical Cost', 'Economic Loss', 'Region', 'Population Density']

# Convert camel case to snake case
def to_snake_case(s):
    # Drop '\' and '/' characters
    s = re.sub(r'[\\/]', '_', s)

    # Replace spaces with underscores
    return re.sub(r'\s+', '_', s).lower()
