import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq
import scipy.stats as stats 
from scipy.stats import mannwhitneyu, wilcoxon, kruskal

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
        
            
        




