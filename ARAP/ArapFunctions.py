import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq
import scipy.stats as stats 
from scipy.stats import mannwhitneyu, wilcoxon, kruskal
import re

import fbprophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

project_id = "cp-gaa-visualization-dev"

import warnings



# Query alternate purchase table in bq

def query_table(sql_query):
    """
    
    """
    
    return pandas_gbq.read_gbq(sql_query, project_id=project_id)
    



def flag_alt_products(data, prod_catalog, column='n1_purchased_product_title'):    
    """ 
    Parses through a product list.
    Identifies if an alternate product is internal or not, and a substitute or not.
    Returns a pandas dataframe.
    
    *************
    Parameter(s):
    *************
    
    1. data: str
        pandas dataframe
        
    2. prod_catalog: str
        A list of unique internal product
       
    3. column: str
        Name of the relevant column.
    """
    toothpaste_catalog = data['product_title'].unique().tolist()
    asin_list = data['asin'].tolist()
    prod_list = data['product_title'].tolist()
    date_list = data['start_date'].tolist()
    alt_purchase_list = data[column].tolist()
    prefix = column.split('_')[0]
    
    internal_list = []
    substitute_list = []
    Yes = 'yes'
    No = 'no'

    for i in alt_purchase_list:
        # Competitors:
            # Case 1
        if i in prod_catalog:
            a1 = Yes
            internal_list.append(a1)
            # Case 2
        else:
            a2 = No
            internal_list.append(a2)

        # Substitutes:
            # Case 1
        if i in toothpaste_catalog:
            b1 = No
            substitute_list.append(b1)
            # Case 2    
        elif ('Toothpaste' not in i) and ('Colgate' in i):
            b2 = No
            substitute_list.append(b2)
            # Case 3    
        elif ('Toothpaste' in i) and ('Colgate' not in i):
            b3 = Yes
            substitute_list.append(b3)
            # Case 4    
        else:
            b4 = No
            substitute_list.append(b4)

    return pd.DataFrame({'start_date': date_list, 
                  'asin': asin_list,
                  'product_title': prod_list, 
                  f'{column}': alt_purchase_list,
                  f'{prefix}_internal': internal_list, 
                  f'{prefix}_substitute': substitute_list}) 



def get_alt_product_count(data, column='n1_purchased_product_title'):
    """
    
    """
    
    alt_prod_counts = data[column].value_counts()
    prefix = column.split('_')[0]
    
    prods_percent = []
    prods_title = []
    for idx in alt_prod_counts.index:
        prods_title.append(idx)

    for j in alt_prod_counts:
        prods_percent.append(j)

    return pd.DataFrame({f'{prefix}_purchased_product_title': prods_title,
                 f'{prefix}_purchased_product_count': prods_percent})

def show_internal_external(data, company=None):
    """
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = data.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Internal', 'External'], minor=True)
    ax.set_ylabel("Percentage")
    plt.title('Share of Alternate Purchases: Internal Product', fontsize=18)
    
    print(counts)

    return plt.show()


def show_substitutes(data):
    """
    
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = data.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['No', 'Yes'])
    ax.set_ylabel("Percentage")
    plt.title('Substitute Products', fontsize=18)
    
    print(counts)

    return plt.show()
    

def filter_by_asin(data):
    """
    filtered by the following asins: ['B01BNEWDFQ', 'B07JWVR1PK', 'B07961C65H', 'B0795VHMM5']
    ['Colgate Cavity Protection Toothpaste with Fluoride - 6 Ounce (Pack of 6)',
    'Colgate Total Whitening Toothpaste - 4.8 Ounce (4 Pack)',
    'Colgate Optic White Whitening Toothpaste, Sparkling White - 5 ounce (3 Pack)',
    'Colgate Optic White High Impact White Whitening Toothpaste, Travel Friendly - 3 Ounce (3 Pack)']
    
    """
    
    return data[(data['asin'] == 'B01BNEWDFQ') \
                     | (data['asin'] == 'B07JWVR1PK')\
                     | (data['asin'] == 'B07961C65H')\
                     | (data['asin'] == 'B0795VHMM5')]
    

def viz_data(data, line=True, scatter_reg=False, scatter_log=False, kde=False):
    """
    
    """
    if line == True:
        
        plt.figure(figsize=(10, 7))
        plt.scatter('lbb_price', 'ordered_revenue', data=data)
        plt.title('Ordered Revenue vs Lost Buy Box', fontsize=16)
        plt.xlabel('Lost Buy Box', fontsize=14)
        plt.ylabel('Ordered Revenue', fontsize=14)
        return plt.show()
    
    elif scatter_reg == True:
        plt.figure(figsize=(10, 7))
        sns.regplot(np.log(data['lbb_price']), np.log(data['ordered_revenue']))
        plt.title('Ordered Revenue vs Lost Buy Box', fontsize=16)
        plt.xlabel('Lost Buy Box', fontsize=14)
        plt.ylabel('Ordered Revenue', fontsize=14)
        return plt.show()
    
    elif scatter_log == True:
        plt.figure(figsize=(10, 7))
        sns.lineplot(np.log(sales_test['lbb_price']), 
             np.log(sales_test['ordered_revenue']), 
             data=sales_test)
        plt.title('Ordered Revenue vs Lost Buy Box (Log-Log Scale)', fontsize=16)
        plt.xlabel('Lost Buy Box', fontsize=14)
        plt.ylabel('Ordered Revenue', fontsize=14)
        return plt.show()
    elif kde == True:
        return sns.jointplot(np.log(sales_test['lbb_price']), 
                             np.log(sales_test['ordered_revenue']), 
                             data= sales_test, kind = 'kde');
    
    
def non_parametric_test(data1, data2, data3, alpha = 0.05, 
                        anova_test=True, mannw=False, h_test=False, wilx=False):
    """
    
    """
    if anova_test == True:
        anova = stats.f_oneway(data1, data2, data3)
        print('ANOVA Test:')
        print('*******')
        if anova.pvalue < alpha:
            return print('Different distribution (reject H0)')
        else:
            return print('Same distribution (fail to reject H0)') 
        
    elif mannw == True:
        ma = mamannwhitneyu(data2, data3)
        print('Mann-Whitney U Test:')
        print('*******')
        if ma.pvalue < alpha:
            return print('Different distribution (reject H0)')
        else:
            return print('Same distribution (fail to reject H0)')
    
    elif h_test == True:
        k_test = kruskal(data1, data2, data3)
        print('Kruskal-Wallis H Test:')
        print('*******')
        if k_test.pvalue < alpha:
            return print('Different distribution (reject H0)')
        else:
            return print('Same distribution (fail to reject H0)')
        
    else:
        wilx = wilcoxon(data2, data3)
        print('Wilcoxon Signed-Rank Test:')
        print('*******')
        if wilx.pvalue < alpha:
            return print('Different distribution (reject H0)')
        else:
            return print('Same distribution (fail to reject H0)')
        
            
        
def flag_alt_products_2(data, prod_catalog, column='n1_purchased_product_title'):    
    """ 
    Parses through a product list.
    Identifies if an alternate product is internal or not, and a substitute or not.
    Returns a pandas dataframe.
    
    *************
    Parameter(s):
    *************
    
    1. data: str
        pandas dataframe
        
    2. prod_catalog: str
        A list of unique internal product
       
    3. column: str
        Name of the relevant column.
    """
    soap_catalog = data['product_title'].unique().tolist()
    asin_list = data['asin'].tolist()
    prod_list = data['product_title'].tolist()
    date_list = data['start_date'].tolist()
    alt_purchase_list = data[column].tolist()
    country_list = data['country'].tolist()
    prefix = column.split('_')[0]
    
    internal_list = []
    substitute_list = []
    Yes = 'yes'
    No = 'no'

    for i in alt_purchase_list:
        # Competitors:
            # Case 1
        if i in prod_catalog:
            a1 = Yes
            internal_list.append(a1)
            # Case 2
        else:
            a2 = No
            internal_list.append(a2)

        # Substitutes:
            # Case 1
        if i in soap_catalog:
            b1 = No
            substitute_list.append(b1)
            # Case 2    
        elif ('Liquid Hand Soap' not in i) and ('Softsoap' in i):
            b2 = No
            substitute_list.append(b2)
            # Case 3    
        elif ('Liquid Hand Soap' in i) and ('Softsoap' not in i):
            b3 = Yes
            substitute_list.append(b3)
            # Case 4    
        else:
            b4 = No
            substitute_list.append(b4)

    return pd.DataFrame({'start_date': date_list, 
                  'asin': asin_list,
                  'product_title': prod_list, 
                  f'{column}': alt_purchase_list,
                  f'{prefix}_internal': internal_list, 
                  f'{prefix}_substitute': substitute_list,
                        'country': country_list}) 


## Use for Softsoap Analysis
def process_data(data, column='n1_purchased_product_title'):
    """
    
    """
    
    prod_catalog = data['product_title'].unique().tolist()
    n1_prod_list = data[column].unique().tolist()
    
    ## Add to the prod_catalog list
    for i in n1_prod_list:
        if ('Softsoap' in i) and (i not in prod_catalog):
            prod_catalog.append(i)
    
    processed_data = flag_alt_products_2(data, prod_catalog)
    return processed_data



def grab_product_size(data):
    """
    
    """
    data_title_1 = data.product_title.tolist()
    final_sizes_1 = []
    temp_1 = []
    
    data_title_2 = data.n1_purchased_product_title.tolist()
    final_sizes_2 = []
    temp_2 = []
    
    for i in data_title_1:
        pattern = re.compile(r'(?:\d).*$')
        size1 = re.findall(pattern, i)
        temp_1.append(size1)

    for j in temp_1:
        if j == []:
            j = 'NA'
            final_sizes_1.append(j)

        else:
            size2 = ''.join(j)
            final_sizes_1.append(size2)
            
   ## second column         
    for i in data_title_2:
        pattern = re.compile(r'(?:\d).*$')
        size3 = re.findall(pattern, i)
        temp_2.append(size3)

    for j in temp_2:
        if j == []:
            j = 'NA'
            final_sizes_2.append(j)

        else:
            size4 = ''.join(j)
            final_sizes_2.append(size4)

    df = pd.DataFrame({'viewed_size': final_sizes_1,
                      'n1_purchased_size': final_sizes_2})
    
    return pd.merge(data, df, left_index=True, right_index=True)



## displays viewed or purchased products
def show_products(data, title='title', date='date'):
    """
    
    """
    counts = data.value_counts()[:11]

    plt.figure(figsize=(28, 20))
    plt.barh(y=counts.index, width=counts)
    plt.xlabel('Number of Views', fontsize=16)
    plt.ylabel('Products', fontsize=16)
    plt.title(f'{title}: {date}', fontsize=24)
    return plt.show()




## displays subplots for viewed and purchased products
def show_products_2(data, title1='title1', title2='title2', date='date'):
    """
    
    """
    
    counts_1 = data.viewed_size.value_counts()[:11]
    counts_2 = data.n1_purchased_size.value_counts()[:11]
    fig = plt.figure(figsize=(28, 20))

    plt.subplot(2, 2, 1)
    plt.barh(y=counts_1.index, width=counts_1)
    plt.ylabel('Products', fontsize=16)
    plt.xlabel('Total Count', fontsize=16)
    plt.title(f"{title1}: {date}", fontsize=20)

    plt.subplot(2, 2, 3)
    plt.barh(y=counts_2.index, width=counts_2)
    plt.ylabel('Products', fontsize=16)
    plt.xlabel('Total Count', fontsize=16)
    plt.title(f"{title2}: {date}", fontsize=20)
    
    return plt.show()






################################################ Times Series Forecasting #############################

# Augmented Dickey-Fuller Test
def interpret_dftest(dftest):
    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic','p-value'])
    return dfoutput


def find_optimal_diff(series):
    """
    finds and returns the lowest difference value d.
    
    Paramter(s):
    1. series: Int
        pandas series
    """
    
    for d in range(1, len(series)):
        print(f'Checking difference of {d}.')
        print(f'p-value = {interpret_dftest(adfuller(series.diff(d).dropna()))["p-value"]}.')

        # If our data, differenced by d time periods, are stationary, print that out!
        if interpret_dftest(adfuller(series.diff(d).dropna()))['p-value'] < 0.05:
            print(f'Differencing our time series by d={d} yields a stationary time series!')
            break

            print()

            
            
            
            
def prep_data_for_tsmodel(data):
    """
    Groups data, sorts values, sets date as index,
    filters by date and Yt, resamples at the daily level,
    and returns a df with renamed columns.
    """
    
    data = data.groupby(['product_title', 'start_date'], 
                   as_index=False)[['shipped_units']].sum()
    data.sort_values('start_date', ascending=True, inplace=True)
    data.set_index('start_date', drop=False, inplace=True)
    
    data = data.loc[:, ['start_date', 'shipped_units']]
    data = data.resample('D').sum()
    data.reset_index(inplace=True)
    
    return data.rename(columns={'start_date': 'ds', 'shipped_units': 'y'})


def forecast_data(data, holidays_df, dates, cap=50000, 
                  cp_prior_scale=0.8, fourier_order=15, periods=180, 
                  added_regressor = False, added_reg_name='added_reg'):
    """
    
    """
    
    if added_regressor == False:
        data['cap'] = cap
        data['covid_period'] = np.where((data['ds'] > dates[0]) & (data['ds'] < dates[1]), True, False)

        model = fbprophet.Prophet(changepoint_prior_scale=cp_prior_scale, yearly_seasonality=False, 
                                       holidays=holidays_df, growth='logistic'
                                      )

        model.add_country_holidays(country_name='US')
        model.add_seasonality(name='yearly', period=30, fourier_order=fourier_order, condition_name='covid_period')
        model.fit(data)

        preds_df = model.make_future_dataframe(periods=periods, freq='D')
        preds_df['cap'] = cap
        preds_df['covid_period'] = np.where((preds_df['ds'] > dates[0]) & (preds_df['ds'] < dates[1]), True, False)

        return model.predict(preds_df), model
    
    else:
        data['cap'] = cap
        data['covid_period'] = np.where((data['ds'] > dates[0]) & (data['ds'] < dates[1]), True, False)

        model = fbprophet.Prophet(changepoint_prior_scale=cp_prior_scale, yearly_seasonality=False, 
                                       holidays=holidays_df, growth='logistic'
                                      )

        model.add_country_holidays(country_name='US')
        model.add_regressor(added_reg_name)
        model.add_seasonality(name='yearly', period=30, fourier_order=fourier_order, condition_name='covid_period')
        model.fit(data)

        preds_df = model.make_future_dataframe(periods=periods, freq='D')
        preds_df['cap'] = cap
        preds_df['covid_period'] = np.where((preds_df['ds'] > dates[0]) & (preds_df['ds'] < dates[1]), True, False)
        preds_df[added_reg_name] = data[added_reg_name]
        preds_df[added_reg_name].fillna(0, inplace=True)

        return model.predict(preds_df), model

def view_compoments(preds, model):
    """
    
    """
    
    return model.plot_components(preds)

    

def view_predictions(y_preds, model, title='', show_v_line=False, start=808):
    """
    Visualize predictions and forecast.
    
    """
    
    if show_v_line == False:
        model.plot(y_preds, xlabel = 'Date', ylabel = 'Shipped Units', figsize=(15, 10))
        plt.title(f'Shipped Units vs Predicted Units: {title}', fontsize=18)
        return plt.show();
    
    else:
        model.plot(y_preds, xlabel = 'Date', ylabel = 'Shipped Units', figsize=(15, 10))
        plt.axvline(y_preds['ds'][start], color='r')
        plt.title(f"Shipped Units vs Predicted Units: {title}", fontsize=18)
        return plt.show();
    
    
def check_model_accuracy(predictions_df, history_df, metric='rmse'):
    """
    Returns model accuracy using a specified metric
    
    Parameter(s):
    1. predictions_df: pandas DataFrame
        dataframe containing predictions from forecast
        
    2. history_df: pandas DataFrame
        dataframe containing original data
        
    3. metric: str
        takes a str with default as 'rmse'. Options include: 'r2', 'mse', 'mae'
    
    
    """
    
    #metric_df = predictions_df.set_index('ds')[['yhat']].join(history_df.set_index('ds').y).reset_index()
    
    a = predictions_df.set_index('ds')[['yhat']]
    b = history_df.set_index('ds').y
    metric_df = a.join(b).reset_index()
    metric_df = metric_df.dropna()
    
    if metric == 'r2':
        return r2_score(metric_df.y, metric_df.yhat)
    
    if metric == 'mse':
        return mean_squared_error(metric_df.y, metric_df.yhat)
    
    if metric == 'mae':
        return mean_absolute_error(metric_df.y, metric_df.yhat)
    
    if metric == 'rmse':
        return np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
    
    
    
    
def return_model_accuracy(predictions_df, history_df):
    """
    Returns a list containing model accuracy metrics
    
    Parameter(s):
    1. predictions_df: pandas DataFrame
        dataframe containing predictions from forecast
        
    2. history_df: pandas DataFrame
        dataframe containing original data
        
    
    
    """
    
    #metric_df = predictions_df.set_index('ds')[['yhat']].join(history_df.set_index('ds').y).reset_index()
    
    a = predictions_df.set_index('ds')[['yhat']]
    b = history_df.set_index('ds').y
    metric_df = a.join(b).reset_index()
    metric_df = metric_df.dropna()
    
    r2 = r2_score(metric_df.y, metric_df.yhat)
    mse = mean_squared_error(metric_df.y, metric_df.yhat)
    mae = mean_absolute_error(metric_df.y, metric_df.yhat)
    rsme = np.sqrt(mean_squared_error(metric_df.y, metric_df.yhat))
    
    return [rsme, r2, mae, mse]

    
    

def grid_search_params(data, changepoints, param_colors, holidays_df, periods=180, freq='D', saturated_growth=False):
    """
    
    """
    if saturated_growth == False:
        #data['covid_period'] = np.where((data['ds'] > '2020-02-29') & (data['ds'] < '2020-04-29'), True, False)
        
        for changepoint in changepoints:
            model = fbprophet.Prophet(changepoint_prior_scale=changepoint)
            #model.add_country_holidays(country_name='US')
            #model.add_seasonality(name='yearly', period=14, fourier_order=10, condition_name='covid_period')
            model.fit(data)

            future = model.make_future_dataframe(periods=periods, freq=freq)
            #future['cap'] = 70000
            #future['covid_period'] = np.where((future['ds'] > '2020-02-29') & (future['ds'] < '2020-04-29'), True, False)
            future = model.predict(future)

            data[changepoint] = future['yhat']


        plt.figure(figsize=(18, 13))
        plt.plot(data['ds'], data['y'], 'ko', label = 'Observations')
        colors = param_colors

        for changepoint in changepoints:
            plt.plot(data['ds'], data[changepoint], color = colors[changepoint], label = '%.3f prior scale' % changepoint)

        plt.legend(prop={'size': 14})
        plt.xlabel('Date'); plt.ylabel('Shipped Units'); plt.title('Effect of Changepoint Prior Scale')
        return plt.show();
    
    else:
        data['cap'] = 70000
        data['covid_period'] = np.where((data['ds'] > '2020-02-29') & (data['ds'] < '2020-04-29'), True, False)
        
        for changepoint in changepoints:
            model = fbprophet.Prophet(changepoint_prior_scale=changepoint, yearly_seasonality=False,
                                     holidays=holidays_df,
                                     growth='logistic')
            model.add_country_holidays(country_name='US')
            model.add_seasonality(name='yearly', period=30, fourier_order=15, condition_name='covid_period')
            model.fit(data)

            future = model.make_future_dataframe(periods=periods, freq=freq)
            future['cap'] = 70000
            future['covid_period'] = np.where((future['ds'] > '2020-02-29') & (future['ds'] < '2020-04-29'), True, False)
            future = model.predict(future)
            data[changepoint] = future['yhat']


        plt.figure(figsize=(18, 13))
        plt.plot(data['ds'], data['y'], 'ko', label = 'Observations')
        colors = param_colors

        for changepoint in changepoints:
            plt.plot(data['ds'], data[changepoint], color = colors[changepoint], label = '%.3f prior scale' % changepoint)

        plt.legend(prop={'size': 14})
        plt.xlabel('Date'); plt.ylabel('Shipped Units'); plt.title('Effect of Changepoint Prior Scale')
        return plt.show(); 
    

def get_holidays(custom_days, year):
    """
    returns a dataframe containg holidays and associated dates in a format the prophet model can understand.

    Argument(s):
    ************

    1. custom_days: dict
        custom dict containing key (dates) and values (holidays)

    2. year: list
        list containing the relevant year(s) in the data 

    **************************
    Example:
    ecomm_days = {'2020-07-06':'Prime Day','2019-11-29': 'Black Friday','2019-12-02': 'Cyber Monday'}
    year = [2019, 2020]

    test_df = get_holidays(ecomm_days, year)
    print(test_df.reset_index(drop=True))

           ds                     holidays
    0   2019-01-01               New Year's Day
    1   2019-01-21  Martin Luther King, Jr. Day
    2   2019-02-18        Washington's Birthday
    3   2019-05-27                 Memorial Day
    4   2019-07-04             Independence Day
    5   2019-09-02                    Labor Day
    6   2019-10-14                 Columbus Day
    7   2019-11-11                 Veterans Day
    8   2019-11-28                 Thanksgiving
    9   2019-11-29                 Black Friday
    10  2019-12-02                 Cyber Monday
    11  2019-12-25                Christmas Day
    12  2020-01-01               New Year's Day
    13  2020-01-20  Martin Luther King, Jr. Day
    14  2020-02-17        Washington's Birthday
    15  2020-05-25                 Memorial Day
    16  2020-07-03  Independence Day (Observed)
    17  2020-07-04             Independence Day
    18  2020-07-06                    Prime Day
    19  2020-09-07                    Labor Day
    20  2020-10-12                 Columbus Day
    21  2020-11-11                 Veterans Day
    22  2020-11-26                 Thanksgiving
    23  2020-12-25                Christmas Day

    """
    import holidays 
    holiday_dict = holidays.US(years=year)
    custom_holiday_dict = holidays.HolidayBase()
    custom_holiday_dict.append(custom_days)

    holiday_list = []
    date_list = []
    for i in holiday_dict.items():
        date_list.append(i[0])
        holiday_list.append(i[1])
    df_a = pd.DataFrame({'ds': date_list, 'holiday': holiday_list})

    custom_holiday_list = []
    custom_date_list = []   
    for i in custom_holiday_dict.items():
        custom_date_list.append(i[0])
        custom_holiday_list.append(i[1])
    df_b = pd.DataFrame({'ds': custom_date_list, 'holiday': custom_holiday_list})

    holiday_df = pd.concat([df_a, df_b])
    holiday_df.reset_index(drop=True)
    return holiday_df.sort_values(by='ds')


def view_predictions2(train_data, test_data, predictions_df, start=100, title='title'):
    """
    
    """
    
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.plot(predictions_df.ds, predictions_df.yhat)
    ax.scatter(train_data.ds, train_data.y, c='black', s=8)
    ax.axvline(predictions_df.ds[start], color='r')
    ax.axvline(test_data.ds.iloc[-1], color='g')
    ax.fill_between(predictions_df.ds, y1=predictions_df.yhat_lower, y2=predictions_df.yhat_upper, alpha=.2)
    ax.scatter(test_data.ds, test_data.y, c='red', s=8)
    plt.xlabel('Date')
    plt.ylabel('Shipped Units')
    plt.title(f'Shipped Units Forecast:\n {title}', fontsize=18)
    plt.grid()
    return fig.show();   