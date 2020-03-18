import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_gbq

import re

import scipy.stats as stats 
from scipy.stats import mannwhitneyu, wilcoxon, kruskal


import warnings


def get_prod_catalog(data):
    """
    Returns a list of all unique products
    
    Parameter(s)
    ************
    
    1. data: str
        pandas dataframe
        
    """
    
    prod_catalog = data['product_title'].unique().tolist()
    print(f'There are {len(prod_catalog)} unique products')
    return prod_catalog

def get_subcategory(data, keyword='Toothpaste'):
    """
    Returns a list filtered by subcategory
    """
    
    sub_catalog = data[data['product_title'].str.contains(keyword)]
    return sub_catalog


class QueryTable:
    """
    class object:
    
    Method(s):
    **********
    
    => query_table
    
    Parameter(s):
    *************
    1. query: str
        sql query
        
    Variable(s):
    ************
    
    1. project_id: str
        bigQuery project id
        
    """
    
    project_id = "cp-gaa-visualization-dev" 
    
    
    def __init__(self, query):
        self.query = query
        
    
    def query_table(self):
        """
        Queries BigQuery table and returns a pandas dataframe
            
        """
        return pandas_gbq.read_gbq(self.query, self.project_id)
        
    


class AlternateProducts:
    """
    class object:
    
    Method(s):
    **********
    
    1. flag_alt_products
    2. set_brand
    3. merge_table
    4. get_alt_product_count
    5. get_competitor_product
    6. show_internal_external
    7. show_substitutes
    
    """
    
    #column = 'n1_purchased_product_title'
    brand = 'Colgate'
    #keyword = 'Toothpaste' 
    
    def __init__(self, data, prod_catalog, column, brand, keyword):
        self.data = data
        self.prod_catalog = prod_catalog
        self.column = column
        self.brand = brand
        self.keyword = keyword
        

    
    def flag_alt_products(self):    
        """ 
        Parses through a product list.
        Identifies if an alternate product is internal or not, and a substitute or not.
        Returns a pandas dataframe.

        """
        
        sub_catalog = self.data['product_title'].unique().tolist()
        asin_list = self.data['asin'].tolist()
        prod_list = self.data['product_title'].tolist()
        date_list = self.data['start_date'].tolist()
        alt_purchase_list = self.data[self.column].tolist()
        prefix = self.column.split('_')[0]

        internal_list = []
        substitute_list = []
        Yes = 'yes'
        No = 'no'

        for i in alt_purchase_list:
            # Internal vs External:
                # Case 1
            if i in self.prod_catalog:
                a1 = Yes
                internal_list.append(a1)
                # Case 2
            else:
                a2 = No
                internal_list.append(a2)

            # Substitutes:
                # Case 1
            if i in sub_catalog:
                b1 = No
                substitute_list.append(b1)
                # Case 2    
            elif (self.keyword not in i) and (self.brand in i):
                b2 = No
                substitute_list.append(b2)
                # Case 3    
            elif (self.keyword in i) and (self.brand not in i):
                b3 = Yes
                substitute_list.append(b3)
                # Case 4    
            else:
                b4 = No
                substitute_list.append(b4)

        return pd.DataFrame({'start_date': date_list, 
                      'asin': asin_list,
                      'product_title': prod_list, 
                      f'{self.column}': alt_purchase_list,
                      f'{prefix}_internal': internal_list, 
                      f'{prefix}_substitute': substitute_list}) 
    

#     @classmethod
#     def set_column(cls, column):
#         cls.column = column
        
    @classmethod
    def set_brand(cls, brand):
        cls.brand = brand
            
        
    @staticmethod
    def merge_table(data_1, data_2, data_3):
        """

        """

        data_2_mask = data_2.loc[:, ['product_title', 'n2_purchased_product_title', 'n2_internal', 'n2_substitute']]
        data_3_mask = data_3.loc[:, ['product_title', 'n3_purchased_product_title', 'n3_internal', 'n3_substitute']]

        merge_1 = pd.merge(data_1, data_2_mask, on='product_title', right_index=True, left_index=True)
        merge_2 = pd.merge(merge_1, data_3_mask, on='product_title', right_index=True, left_index=True)

        return merge_2

    @staticmethod
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
    
    @staticmethod
    def get_competitor_product(data, column = 'n1_purchased_product_title', keyword = 'Crest'):
        """
        
        """
        
        return data[data[column].str.contains(keyword)]
    
    @staticmethod
    def show_internal_external(data):
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

    @staticmethod
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
    
    @staticmethod
    def show_internal_external_subplot(data1, data2, data3):
        """


        """

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

        sns.barplot(data1.value_counts().index, y = data1.value_counts(normalize=True)*100, ax=ax1)
        ax1.set_xticklabels(['Internal', 'External'], minor=True)
        ax1.set_ylabel('Percentage')
        ax1.set_title('Frequently Purchased 1')

        sns.barplot(data2.value_counts().index, y = data2.value_counts(normalize=True)*100, ax=ax2)
        ax2.set_xticklabels(['Internal', 'External'], minor=True)
        ax2.set_ylabel('')
        ax2.set_title('Frequently Purchased 2')

        sns.barplot(data3.value_counts().index, y = data3.value_counts(normalize=True)*100, ax=ax3)
        ax3.set_xticklabels(['Internal', 'External'], minor=True)
        ax3.set_ylabel('')
        ax3.set_title('Frequently Purchased 3')


        fig.suptitle('Share of Alternate Purchases: Internal Product', fontsize=18)
        
        print(data1.value_counts(normalize=True)*100)
        print('************\n')
        print(data2.value_counts(normalize=True)*100)
        print('************\n')
        print(data3.value_counts(normalize=True)*100)

        return plt.show()
    
    @staticmethod
    def show_substitues_subplot(data1, data2, data3):
        """


        """

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

        sns.barplot(data1.value_counts().index, y = data1.value_counts(normalize=True)*100, ax=ax1)
        ax1.set_xticklabels(['No', 'Yes'], minor=True)
        ax1.set_ylabel('Percentage')
        ax1.set_title('Frequently Purchased 1')

        sns.barplot(data2.value_counts().index, y = data2.value_counts(normalize=True)*100, ax=ax2)
        ax2.set_xticklabels(['No', 'Yes'], minor=True)
        ax2.set_ylabel('')
        ax2.set_title('Frequently Purchased 2')

        sns.barplot(data3.value_counts().index, y = data3.value_counts(normalize=True)*100, ax=ax3)
        ax3.set_xticklabels(['No', 'Yes'], minor=True)
        ax3.set_ylabel('')
        ax3.set_title('Frequently Purchased 3')


        fig.suptitle('Substitute Product', fontsize=18)
        
        print(data1.value_counts(normalize=True)*100)
        print('************\n')
        print(data2.value_counts(normalize=True)*100)
        print('************\n')
        print(data3.value_counts(normalize=True)*100)

        return plt.show()
    
    def flag_alt_hills_products(self):
        """
        
        """
        
        asin_list = self.data['asin'].tolist()
        prod_list = self.data['product_title'].tolist()
        date_list = self.data['start_date'].tolist()
        alt_purchase_list = self.data[self.column].tolist()
        alt_catalog = self.data[self.column].unique().tolist()
        prefix = self.column.split('_')[0]

        internal_list = []
        external_list = []
        substitute_list = []
        sub_external_list = []
        Yes = 'yes'
        No = 'no'



        alt_purchase_list_lower = [i.lower() for i in alt_purchase_list]
        prod_catalog_lower = [i.lower() for i in self.prod_catalog]

        for i in alt_purchase_list_lower:
            x = re.findall("(hill's)|(dog food)|(cat food)|(dog treats)|(cat treats)|(wet dog food)|(wet cat food)|(dry dog food)|(dry cat food)|(dry dog treats)|(dry cat treats)|(wet dog treats)|(wet cat treats)|(chews)", i)
            sub_external_list.append(x)

        for i in alt_purchase_list_lower:
            # Competitors:
                # Case 1
            if i in prod_catalog_lower:
                a1 = Yes
                internal_list.append(a1)
                # Case 2
            else:
                a2 = No
                internal_list.append(a2)

            # Substitutes:
                # Case 2       
        for i in sub_external_list:
            if i == []:
                b1 = No
                substitute_list.append(b1)
            elif "hill's".lower() in i[0]:
                b2 = No
                substitute_list.append(b2)
            else:
                b3 = Yes
                substitute_list.append(Yes)

        return pd.DataFrame({'start_date': date_list, 
                      'asin': asin_list,
                      'product_title': prod_list, 
                      f'{self.column}': alt_purchase_list,
                      f'{prefix}_internal': internal_list, 
                      f'{prefix}_substitute': substitute_list}) 


