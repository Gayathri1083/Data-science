# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:00:55 2020

@author: Gayathri
"""

## loading the data
import pandas as pd
import matplotlib.pyplot as plt

housingdata = pd.read_excel("https://data.london.gov.uk/download/uk-house-price-index/70ac0766-8902-4eb5-aab5-01951aaed773/UK%20House%20price%20index.xls",sheetname='Average price',index_col=None)

##examining the data
housingdata.head(5)

## transpose the dataframe as we need the rows to represent the cities of London
housingdata = housingdata.transpose()

##reset index as we want the index to be unique
housingdata = housingdata.reset_index()

##drop the three rows that contain NAN values
housingdata = housingdata.dropna(how="any")

##rename the city code column
housingdata = housingdata.rename(columns={pd.NaT:"city code"})
housingdata = housingdata.rename(columns={"index":"city name"})

##need to reset index again as we have dropped some rows
housingdata = housingdata.reset_index(drop=True)

##make the dates in one column and the values in another column rathar than having as many columns
housingdata = pd.melt(housingdata,id_vars=['city name','city code'])


##rename columns to be meaningful
housingdata = housingdata.rename(columns={'variable':'Date','value':'Avg_price'})

##check the datatype of housingdata
housingdata.dtypes

##convert the avg_price to numeric
housingdata['Avg_price'] = pd.to_numeric(housingdata['Avg_price'])

##recheck the datatype
housingdata.dtypes

##find our the year of each data separately
housingdata['Year'] = housingdata['Date'].apply(lambda x:x.year)

##drop the date column as we coneverted to year
del housingdata['Date']

##group by city name, year - the average price for each year
housingdata = housingdata.groupby(by = ['city name','Year']).mean()

##reset the index
housingdata = housingdata.reset_index()

##create function to calculate the price ratio
def createratio(df):
    year1995 = float(df['Avg_price'][df['Year'] == 1995])
    year2020 = float(df['Avg_price'][df['Year'] == 2020])
    ratio = [year2020/year1995]
    return ratio

ratioprice = {}

##calculate the price ratio for each city
for i in housingdata['city name']:
    housingdatabycityname = housingdata[housingdata['city name'] == i]
    ratioprice[i] = createratio(housingdatabycityname) 
print(ratioprice)    

## keep it as a dataframe
df = pd.DataFrame(ratioprice)

##transpose the dataframe
df = df.transpose()

##reset index
df = df.reset_index()

##rename the columns
df = df.rename(columns = {'index':'city name',0:'ratio'})

##sort the ratio in descending order 
df = df.sort_values(['ratio'], ascending=[0])

##Visualize the first 10 top cities
top10 = df.head(10)
plt = top10.plot(x='city name',y='ratio',rot=90)










    


