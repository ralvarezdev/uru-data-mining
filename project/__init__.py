import os
import re

# Constants
DATASET_DIR = 'dataset'
ROAD_ACCIDENTS_DIR = 'road-accidents'
ROAD_ACCIDENT_DATASET_FILE = 'road_accident_dataset.csv'
TRANSFORMED_ROAD_ACCIDENT_DATASET_FILE = 'transformed_road_accident_dataset.csv'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTOGRAMS_DIR = 'histograms'
BOXPLOTS_DIR='boxplots'
COUNTPLOTS_DIR='countplots'
OLS_DIR= 'ols'
PAIRPLOTS_DIR='pairplots'
RIDGE_REGRESSIONS_DIR='ridge_regressions'
RIDGE_REGRESSIONS_FILE='ridge_regressions.txt'
LASSO_REGRESSIONS_DIR='lasso_regressions'
LASSO_REGRESSIONS_FILE='lasso_regressions.txt'
ELASTIC_NET_REGRESSIONS_DIR='elastic_net_regressions'
ELASTIC_NET_REGRESSIONS_FILE='elastic_net_regressions.txt'
PREVIEW_ROWS = 10
COLUMNS_CHUNK_SIZE = 5
COLUMNS = ['Country', 'Year', 'Month', 'Day of Week', 'Time of Day', 'Urban/Rural', 'Road Type', 'Weather Conditions', 'Visibility Level', 'Number of Vehicles Involved', 'Speed Limit', 'Driver Age Group', 'Driver Gender', 'Driver Alcohol Level', 'Driver Fatigue', 'Vehicle Condition', 'Pedestrians Involved', 'Cyclists Involved', 'Accident Severity', 'Number of Injuries', 'Number of Fatalities', 'Emergency Response Time', 'Traffic Volume', 'Road Condition', 'Accident Cause', 'Insurance Claims', 'Medical Cost', 'Economic Loss', 'Region', 'Population Density']

# Convert camel case to snake case
def to_snake_case(s):
    # Drop '\' and '/' characters
    s = re.sub(r'[\\/]', '_', s)

    # Replace spaces with underscores
    return re.sub(r'\s+', '_', s).lower()
