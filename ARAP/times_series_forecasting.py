import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

%matplotlib inline

import warnings
warnings.simplefilter(action="ignore")

import fbprophet
import holidays

from ArapFunctions import query_table, prep_data_for_tsmodel, view_predictions, check_model_accuracy, return_model_accuracy

from arap_helpers import QueryTable





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




class forecaster:
    
    condition_dates = ['2020-02-29', '2020-04-29']
    cap = 50000
    
    
    def __init__(self, data, model, 
                 periods=180, freq='D', holidays=None, 
                 added_regressor=None, added_reg_name='reg_name'):
        
        self.data = data
        self.model = model
        self.periods = periods
        self.freq = freq
        self.added_regressor = added_regressor
        self.added_reg_name = added_reg_name
        
        
    @classmethod
    def set_new_cap(cls, new_cap):
        cls.cap = new_cap
        
    def build_future_dataframe(self):
        if self.added_regressor == None:
            future = self.model.make_future_dataframe(periods=self.periods, freq=self.freq)
            future['cap'] = self.cap
            return future
        
        else:
            future = self.model.make_future_dataframe(periods=self.periods, freq=self.freq)
            future['cap'] = self.cap
            return future
            
    def seasonal_effects(self, name='yearly', period=30, fourier_order=15, condition_name='covid_period'):
        self.data[condition_name] = np.where((self.data['ds'] > self.condition_dates[0]) & (self.data['ds'] < self.condition_dates[1]), True, False)
        self.model.add_seasonality(name=name, period=period, fourier_order=fourier_order, condition_name=condition_name)
        
    def add_regressor(self, condition_name='covid_period'):
        self.data[condition_name] = np.where((data['ds'] > self.condition_dates[0]) & (self.condition_dates['ds'] < dates[1]), True, False)
        self.model.add_regressor(self.added_reg_name)      
           
    def fit_model(self):
        self.model.fit(self.data)
    
    def predict(self, condition_name='covid_period'):
        global y_preds
        
        if self.added_regressor == None:
            y_preds = self.model.make_future_dataframe(periods=self.periods, freq=self.freq)
            y_preds['cap'] = self.cap
            y_preds[condition_name] = np.where((y_preds['ds'] > self.condition_dates[0]) & (y_preds['ds'] < self.condition_dates[1]), True, False)  
            y_preds = self.model.predict(y_preds)
            return y_preds
        
        else:
            y_preds = self.model.make_future_dataframe(periods=self.periods, freq=self.freq)
            y_preds['cap'] = self.cap
            y_preds[condition_name] = np.where((y_preds['ds'] > self.condition_dates[0]) & (y_preds['ds'] < self.condition_dates[1]), True, False)  
            y_preds[self.added_reg_name] = self.data[self.added_reg_name]
            y_preds[self.added_reg_name].fillna(0, inplace=True)
            y_preds = self.model.predict(y_preds)
            return y_preds
    
    
    def plot_predictions(self, show_v_line=True, title='title', condition_name='covid_period'):
        start=len(self.data)
        
        if show_v_line == True: 
            self.model.plot(y_preds, xlabel = 'Date', ylabel = 'Shipped Units', figsize=(15, 10))
            plt.axvline(y_preds['ds'][start], color='r')
            plt.title(f"Shipped Units vs Predicted Units: {title}", fontsize=18)
            return plt.show();
        
        else:
            self.model.plot(y_preds, xlabel = 'Date', ylabel = 'Shipped Units', figsize=(15, 10))
            plt.title(f"Shipped Units vs Predicted Units: {title}", fontsize=18)
            return plt.show();
        
    def plot_predictions2(self, test_data, title='title'):
        start=len(self.data)
        fig, ax = plt.subplots(figsize=(18, 13))
        ax.plot(y_preds.ds, y_preds.yhat)
        ax.scatter(self.data.ds, self.data.y, c='black', s=8)
        ax.axvline(y_preds.ds[start], color='r')
        ax.axvline(test_data.ds.iloc[-1], color='g')
        ax.fill_between(y_preds.ds, y1=y_preds.yhat_lower, y2=y_preds.yhat_upper, alpha=.2)
        ax.scatter(test_data.ds, test_data.y, c='red', s=8)
        plt.xlabel('Date')
        plt.ylabel('Shipped Units')
        plt.title(f'Shipped Units Forecast:\n {title}', fontsize=18)
        plt.grid()
        return fig.show();  
