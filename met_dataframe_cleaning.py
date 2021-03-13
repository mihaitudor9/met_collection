import matplotlib
from tabulate import tabulate
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import operator
import functools
import re
pd.set_option('mode.chained_assignment', None)

# Due to mixed values, we will be treating most fields as text for now and see if there
# is anything to be done such that numerical values could be extracted out of these.
# There are however a few Boolean and Numerical types exceptions.

df = pd.read_csv(r"C:\Users\Tudor\PycharmProjects\Intelligent\MetObjects.txt", dtype={
    'Gallery Number': str, 'Period': str,
    'Portfolio': str, 'AccessionYear': str,
    'River': str, 'Subregion': str,
    'Dynasty': str, 'Excavation': str,
    'Locale': str, 'Locus': str, 'Reign': str,

}, low_memory=False, nrows=9980)

# Reading all the rows seems to be taking quite a long time
#}, low_memory = False)

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

pd.set_option('display.max_columns', None)
df.loc[:,'Covered Area'] = 0
df.loc[:,'Covered Volume'] = 0

# print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

print(df.dtypes)

df['Department'].value_counts().plot(kind='bar')
plt.title("Counts for items in each one of the departments")
plt.xlabel("Department")
plt.ylabel("Number of stored items")
plt.show()

df = df[df['Object Number'].notna()]
df = df[df['Dimensions'].notna()]
df = df[df['Object ID'].notna()]

# Originally, the dimensions of each object are stored both in inches and centimeters
# Should only keep the centimeters dimensions
# i.e. 15 3/4 x 11 3/4 in. (40 x 29.8 cm) - > 40 x 29.8
df['Dimensions'] = df['Dimensions'].apply(lambda st: st[st.find("(") + 1:st.find(")")])
df['Dimensions'] = (df['Dimensions'].str.replace("cm", ""))
df['Dimensions'] = (df['Dimensions'].str.replace(" ", ""))
df['Dimensions'] = (df['Dimensions'].str.replace("×", "x"))

# We can see that some of the items do not have a specified dimension, would rather delete
# those rows since there's not much we can do with badly catalogued items like these.
df = df[~df['Dimensions'].str.contains("unavailabl", na = False)]

# For now, it might not be the best idea to turn the <Dimensions> column into numerical types
# since there is a mixture of 2-dimensional and 3-dimensional items (e.g. 1 x 5.3 VS 2 x 4 x 8)
df['Dimensionality'] = df['Dimensions'].apply(lambda st: str.count(st, "x") + 1)

# Some items are being characterized by the diameter instead of its length or width
# We need to standardize the dimensions:
# 1.5 - > 1.5x1.5
for index, (dimensionality, dimensions) in enumerate(zip(df['Dimensionality'], df['Dimensions'])):
    if(df.iloc[index]['Dimensionality']) == 1:
        df.at[index,'Dimensionality'] = 2
        df.at[index,'Dimensions'] = df.iloc[index]['Dimensions'] + "x" + df.iloc[index]['Dimensions']

df['Object ID'] = df['Object ID'].astype('Int64')
df['Object Begin Date'] = df['Object Begin Date'].astype('Int64')
df['Object End Date'] = df['Object End Date'].astype('Int64')
df['Dimensionality'] = df['Dimensionality'].fillna(0).astype('Int64')

"""
# I'm pretty sure this can be done in a more efficient way but still
for index, (dimensionality, dimensions) in enumerate(zip(df['Dimensionality'], df['Dimensions'])):

        number_dimensions = []
        text_dimension = df.iloc[index]['Dimensions']
        number_dimensions = [float(s) for s in re.findall(r'-?\d+\.?\d*', text_dimension)]
        print(text_dimension, number_dimensions)

        if(len(number_dimensions) == 2):
            covered_area = np.prod(np.array(number_dimensions))
            covered_area = round(covered_area,2)
            covered_volume = 0

        elif (len(number_dimensions) == 3):
            covered_area = 0
            covered_volume = np.prod(np.array(number_dimensions))
            covered_volume = round(covered_volume,2)

        df.loc[index, 'Covered Area'] = covered_area
        df.loc[index, 'Covered Volume'] = covered_volume

        print("Index: ", index)
        print("Covered Area: ", covered_area)
        print("Covered Volume:" , covered_volume)
        print("---------------")

"""

# Several field support multiple values, they should look similarly to the values
# at the <Tags> field in order to reduce the confusion.
df['Medium'] = df['Medium'].str.lower()
df['Medium'] = (df['Medium'].str.replace(", ", "|"))
df['Artist Nationality'] = (df['Artist Nationality'].str.replace(", ", "|"))

"""
    Artist Display Name    Artist Display Bio 
102         James Davis        active 1803–28       
103       John Molineux  active ca. 1800–1820   
104       John Molineux  active ca. 1800–1820   
105      Revere and Son            ca. 1787–?    
106      Revere and Son            ca. 1787–?    

For example, I find the <Artist Display Bio> quite useless. Do we care during which years was the artist
active ? It's more likely we are interested in the years the artist lived. 
"""
del df['Artist Display Bio']

"""
                Object Date       Object Begin Date        Object End Date  \
102                1803–28               1803                    1828   
103                ca. 1810              1807                    1810   
104                ca. 1810              1807                    1810   
105                1787–1810             1787                    1810   
106                1787–1810             1787                    1810     
There's some sort of redundancy in the <Object Date> field
"""
del df['Object Date']

# The object number is a string stored like it would be some sort of IP address
# df['Object Number'] = (df['Object Number'].str.split()).apply(lambda x: (x[0].replace('.', '')))

# There is a "Primary Key" <Object ID> that has the same scope as <Object Number>
del df['Object Number']

# We already have the <Region> field so this seems like an overkill
del df['Subregion']

# Re-order the columns such that <Object ID> is the first column.
# Other reorders should follow
df = df[['Object ID', 'Is Highlight', 'Is Timeline Work', 'Department', 'Object Name', 'Title', 'Culture', 'Period',
         'Dynasty', 'Reign', 'Is Public Domain', 'Gallery Number', 'AccessionYear',
         'Portfolio', 'Constiuent ID', 'Artist Display Name',
         'Artist Nationality', 'Artist Gender', 'Artist Wikidata URL',
         'Object Begin Date', 'Object End Date', 'Medium', 'Dimensionality', 'Dimensions','Covered Area','Covered Volume', 'Credit Line',
         'City', 'State', 'Country', 'Region', 'Locale', 'Locus', 'Excavation', 'River',
         'Classification', 'Link Resource',
         'Object Wikidata URL', 'Repository', 'Tags']]

print(df.head(30))



