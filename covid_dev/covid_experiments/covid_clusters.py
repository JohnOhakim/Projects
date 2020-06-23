import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go


import warnings
warnings.filterwarnings('ignore')


from sklearn.mixture import GaussianMixture

import clustering as cl


pos_data = pd.read_csv('./data/pos_and_external.csv')
annotations_df = pd.read_csv('./data/COVID-19_US_Tiimeline_4.csv')
annotations_df['Date'] = pd.to_datetime(annotations_df.Date)

features = ['lift', 'LegacyStringencyIndex']
gmm = GaussianMixture(covariance_type='diag', n_components=6, random_state=200507)

cl.plot_clusters2(pos_data, 'United States', 'Liquid Hand Wash', annotations_df, gmm, features)

