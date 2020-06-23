import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas_gbq

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import hdbscan
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from yellowbrick.cluster import KElbowVisualizer

import clustering as cl


project_id = "cp-saa-dev-covid19"


def query_table(sql_query):
    return pandas_gbq.read_gbq(sql_query, project_id=project_id)


def import_data(path):
    return pd.read_csv(path)

def calculate_lift(data, kpi):
    data["lift"] = (data[kpi] - data["baseline"]) / data["baseline"]
    return data

def build_annotations(df, country):

    annotations = []
    y = 1
    for _, row in df.iterrows():
        annotations.append(dict(
            x=row["Date"],
            y=y,
            xref="x2",
            yref="y2",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-20 * y,
            bgcolor='white',
            text=row["Event"],
            hovertext=row["Description"]
        ))
        y *= -1
    #print(annotations)
    annotations = annotations[0:2] + annotations

    if country == "United States":
        annotations[7]["ay"] += 30

    return annotations




def filter_data(data, country, subcat):
    return data[(data['country'] == country) & (data['subcategory'] == subcat)]

def scale_matrix(X):
    pt = PowerTransformer()
    return pt.fit_transform(X)

def cluster_X(df, features, algorithm):
    X = df[features]
    X_scaled = scale_matrix(X)
    X_mean = np.nanmean(X_scaled, axis=0)
    idx = np.where(np.isnan(X_scaled))
    X_scaled[idx] = np.take(X_mean, idx[1])
    
    algorithm.fit(X_scaled)
    print(f"Silhouette Score: {silhouette_score(X_scaled, algorithm.labels_)}")
    
    df['clusters'] = algorithm.labels_
    #df['cluster_membership_score'] = algorithm.probabilities_
    return df

def cluster_X2(df, features, algorithm):
    X = df[features]
    X_scaled = scale_matrix(X)
    X_mean = np.nanmean(X_scaled, axis=0)
    idx = np.where(np.isnan(X_scaled))
    X_scaled[idx] = np.take(X_mean, idx[1])
    
    algorithm.fit(X_scaled)
    
    labels = algorithm.predict(X_scaled)
    new_df = pd.DataFrame(df)
    new_df['clusters'] = labels
    return new_df



def plot_clusters(data, country, subcat, annotations_df, kpi='qty_sold'):
    """
    
    """
    
    df = data.copy()
    df['lift_percent'] = df['lift'].apply(lambda x: x * 100)
    df['lift_color'] = np.where(df['lift_percent'] > 0, 'green', 'red')
    
    new_df = filter_data(df, country, subcat)

    fig = make_subplots(rows=2, cols=1)

    fig.add_trace(
        go.Bar(x=new_df.date, y=new_df[kpi], 
               name = f"{kpi}",
               text=new_df['clusters'].apply(lambda x: str(x)), 
               hoverinfo='x+y+text',
               marker=dict(color=new_df.clusters, showscale=True)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=new_df.date, y=new_df.baseline, 
                   name= 'Baseline', mode='lines'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=new_df.date, y=new_df.lift_percent,
              name= '% Lift', marker=dict(color=new_df.lift_color)),
        row=2, col=1
    )

    fig.update_layout(height=600, title_text=f"{country} Consumer Behavior: {subcat}", legend_orientation="h", 
                     annotations=build_annotations(annotations_df, f'{country}')
    )
    return fig.show()


def show_optimal_clusters(data, features, algorithm, metric='distortion'):
   
    visualizer = KElbowVisualizer(algorithm, k=(2,12), metric=metric)


    X = data[features]
    X_scaled = cl.scale_matrix(X)
    X_mean = np.nanmean(X_scaled, axis=0)
    idx = np.where(np.isnan(X_scaled))
    X_scaled[idx] = np.take(X_mean, idx[1])

    visualizer.fit(X_scaled)       
    return visualizer.show();


def select_best_gmm(X_scaled):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type=cv_type)
            gmm.fit(X_scaled)
            bic.append(gmm.bic(X_scaled))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    return best_gmm


def cluster_X3(df, features, algorithm, subcat):
    
    temp_df = df[df['subcategory'] == subcat]
    X = temp_df[features]
    X_scaled = scale_matrix(X)
    X_mean = np.nanmean(X_scaled, axis=0)
    idx = np.where(np.isnan(X_scaled))
    X_scaled[idx] = np.take(X_mean, idx[1])
    
    algorithm.fit(X_scaled)
    
    labels = algorithm.predict(X_scaled)
    new_df = pd.DataFrame(temp_df)
    new_df['clusters'] = labels
    return new_df


def plot_clusters2(data, country, subcat, annotations_df, algorithm, features, kpi='qty_sold'):
    """
    
    """
    
    df = data.copy()
    temp_df = cluster_X3(df, features, algorithm, subcat)
    
    temp_df['lift_percent'] = temp_df['lift'].apply(lambda x: x * 100)
    temp_df['lift_color'] = np.where(temp_df['lift_percent'] > 0, 'green', 'red')
    temp_df['cluster_str'] = temp_df['clusters'].apply(lambda x: str(x))
        
    new_df = filter_data(temp_df, country, subcat)
        
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    fig.add_trace(
        go.Bar(x=new_df.date, y=new_df[kpi], 
               name = f"{kpi}",
               text=new_df['clusters'].apply(lambda x: str(x)), 
               hoverinfo='x+y+text',
               marker=dict(color=new_df.clusters, showscale=True, colorbar=dict(title='Clusters'))), 
               
               row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=new_df.date, y=new_df.baseline, 
                   name= 'Baseline', mode='lines'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=new_df.date, y=new_df.lift_percent,
              name= '% Lift', marker=dict(color=new_df.lift_color)),
        row=2, col=1
    )

    fig.update_layout(height=600, title_text=f"{country} Consumer Behavior: {subcat}", legend_orientation="h", 
                     annotations=build_annotations(annotations_df, f'{country}')
    )
    return fig.show()
