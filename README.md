# Market-Basket-Analysis-Project
# PROJECT 1 by Aribisogan Oluwaseun

# Market Basket Analysis for E-Commerce Notebook

# Introduction 
Market Basket Analysis is a useful technique for companies that want to optimize their product offers, boost cross-selling opportunities, and improve their marketing strategy. It can result in increased income, improved customer happiness, and overall business success.As a data analyst,I have been provided a dataset to work with, and the main thing is to use market Basket Analysis to uncover patterns in customer purchasing behavior.

Below is the process:

Loading the data
Simple EDA and an example of feature engineering
Data preprocessing and data wrangling
Data Visualization

# 1. Import Important Libraries

#Importing important libraries

#Data Analysis and Data Wrangling
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
%matplotlib inline

#Import Apriori Algorithm
from mlxtend.frequent_patterns import apriori, association_rules

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

## 2. Import the dataset

#Loading the dataset
data = pd.read_csv('Market Basket Analysis - Groceries_dataset (2).csv')

#Seeing what our data looks like.
data.head()

#Checking the number of rows and columns.
print(f"The dataset has {data.shape[0]} rows, and {data.shape[1]} columns")

#Let'see the datatypes of our variables
data.info()

#Checking the number of null values in our dataset
data.isnull().sum()


#let’s have a look at the summary statistics of this dataset:
data.describe()

## 3: Explore Target distribution 


sns.catplot(x="itemDescription", kind="count", data=data)

## let’s have a look at the sales distribution of items:

fig = px.histogram(data, x='itemDescription', 
                   title='Item Distribution')
fig.show()

# Now, let’s have a look at the top 10 most popular items picked by their unique number:

# Calculate item popularity
item_pop = data.groupby('itemDescription')['Member_number'].sum().sort_values(ascending=False)

top_n = 10
fig = go.Figure()
fig.add_trace(go.Bar(x=item_pop.index[:top_n], y=item_pop.values[:top_n],
                     text=item_pop.values[:top_n], textposition='auto',
                     marker=dict(color='blue')))
fig.update_layout(title=f'Top {top_n} Most Popular Items',
                  xaxis_title='Item Name', yaxis_title='Unique Num')
fig.show()

Here I explore the top 10 item popularity,comparing with the member number using the chart above.

## Now, let’s use the Apriori algorithm to create association rules. The Apriori algorithm is used to discover frequent item sets in large transactional datasets. It aims to identify items that are frequently purchased together in transactional data. It helps uncover patterns in customer behaviour, allowing businesses to make informed decisions about product placement, promotions, and marketing. Here’s how to implement Apriori to generate association rules:

#Convert the dataset into transactional format
transactions = data.groupby(['Date'])['itemDescription'].apply(list)
transactions

# 4.Creating Dummy Variables

#I am going to create a one-hot matrix of the items
one_hot = pd.get_dummies(data['itemDescription'])
one_hot

#Add the Date column back to the one-hot encoded matrix
one_hot['Date']=data['Date']
one_hot

#Now, we group the One-Hot Matrix by date and sum the values
one_hot = one_hot.groupby('Date').sum()
one_hot

#Now, we merge the one-hot encoded matrix, with the transactional data
transaction_matrix = pd.merge(transactions, one_hot, on='Date')
transaction_matrix

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
print(transaction_matrix.dtypes)

from mlxtend.frequent_patterns import apriori, association_rules

# Group items by Date and create a list of items for each date
basket = data.groupby('Member_number')['itemDescription'].apply(list).reset_index()

# Encode items as binary variables using one-hot encoding
basket_encoded = basket['itemDescription'].str.join('|').str.get_dummies('|')

# Find frequent itemsets using Apriori algorithm with lower support
frequent_itemsets = apriori(basket_encoded, min_support=0.01, use_colnames=True)

# Generate association rules with a lower lift threshold
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.5)

# Display association rules
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

Antecedents: These items are considered the starting point or “if” part of the association rule. For example, Beef, Bottled water, brown bread, and butter are the antecedents in this analysis.

Consequents: These items tend to be purchased along with the antecedents or the “then” part of the association rule.

Support: Support measures how frequently a particular combination of items (both antecedents and consequents) appears in the dataset. It is the proportion of transactions in which the items are bought together. For example, the first rule indicates that Beef and UHT milk are bought together in approximately 1.05% of all transactions.

Confidence: Confidence quantifies the likelihood of the consequent item being purchased when the antecedent item is already in the basket. In other words, it shows the probability of buying the consequent item when the antecedent item is bought. For example, the first rule tells us that there is an 87.98% chance of buying UTH milk when beef is already in the basket.

Lift: Lift measures the degree of association between the antecedent and consequent items, while considering the baseline purchase probability of the consequent item. A lift value greater than 1 indicates a positive association, meaning that the items are more likely to be bought together than independently. A value less than 1 indicates a negative association. For example, the first rule has a lift of approximately 1.12, suggesting a positive association between Beef and UTH milk.


