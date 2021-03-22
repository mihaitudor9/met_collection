import matplotlib
# from tabulate import tabulate
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import re
from geopy.geocoders import Nominatim
import seaborn as sns
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
import folium
from folium.plugins import MarkerCluster

sns.set_theme(style="darkgrid")
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

# Due to mixed values, we will be treating most fields as text for now and see if there
# is anything to be done such that numerical values could be extracted out of these.
# There are however a few Boolean and Numerical types exceptions.

df = pd.read_csv(r"C:\Users\Tudor\PycharmProjects\Intelligent\MetObjects.txt", dtype={
    'Gallery Number': str, 'Period': str,
    'Portfolio': str,
    'Dimensions': str

}, low_memory=False, nrows=20000)

# We are only interested in the paintings and drawings
df = df.loc[df['Classification'].isin(['Paintings', 'Drawings'])]

# Re-order the columns such that <Object ID> is the first column.
# Other reorders should follow
df = df[
    ['Object ID', 'Department', 'AccessionYear', 'Object Name', 'Title', 'Culture',
     'Reign', 'Artist Display Name', 'Artist Nationality', 'Artist Begin Date', 'Artist End Date',
     'Artist Wikidata URL',
     'Object Begin Date', 'Object End Date', 'Medium', 'Dimensions',
     'City', 'Country', 'Classification', 'Object Wikidata URL', 'Tags']]

# Number of rows that have a value NaN
for column in df:
    print(column, len(df[pd.isnull(df[column])]), "/", len(df['Object ID']))

# In case we can fill the NaN values with 0
# df1 = df.copy()
# Fill all NaN values with 0

# Displaying all values of the dataframe if necessary
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Displaying all unique values in every column
# for col in df:
# print(col, df[col].unique())

# Clean up of the Culture column
df['Culture'] = df['Culture'].str.replace('for export', '')
df['Culture'] = df['Culture'].str.replace(', for Swedish market', ' (Swedish market)')
df['Culture'] = df['Culture'].str.replace(', for American market', ' (American market)')
df['Culture'] = df['Culture'].str.replace('or', ',')
df['Culture'] = df['Culture'].str.replace(', probably', '')
df['Culture'] = df['Culture'].str.replace(', possibly', '')
df['Culture'] = df['Culture'].str.replace('probably', '')

df.loc[:, 'Covered Area'] = -1
# print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
# print(df.dtypes)

df = df[df['Object ID'].notna()]
df = df[df['Dimensions'].notna()]

# Several field support multiple values, they should look similarly to the values
# at the <Tags> field in order to reduce the confusion.
df['Medium'] = df['Medium'].str.lower()
df['Medium'] = (df['Medium'].str.replace(", ", "|"))
df['Artist Nationality'] = (df['Artist Nationality'].str.replace(", ", "|"))

# Originally, the dimensions of each object are stored both in inches and centimeters
# Should only keep the centimeters dimensions
# i.e. 15 3/4 x 11 3/4 in. (40 x 29.8 cm) - > 40 x 29.8
# keeping in mind that the resulting column will denote the dimensions in cm
df['Dimensions'] = df['Dimensions'].apply(lambda st: st[st.find("(") + 1:st.find(")")])
df['Dimensions'] = (df['Dimensions'].str.replace("cm", ""))
df['Dimensions'] = (df['Dimensions'].str.replace(" ", ""))
df['Dimensions'] = (df['Dimensions'].str.replace("×", "x"))

# For now, it might not be the best idea to turn the <Dimensions> column into numerical types
# since there is a mixture of 2-dimensional and 3-dimensional items (e.g. 1 x 5.3 VS 2 x 4 x 8)
df['Dimensionality'] = df['Dimensions'].apply(lambda st: str.count(st, "x") + 1)

for index, (dimensionality, dimensions) in enumerate(zip(df['Dimensionality'], df['Dimensions'])):
    if (df.iloc[index]['Dimensionality']) == 1:
        df.at[index, 'Dimensionality'] = 2
        df.at[index, 'Dimensions'] = df.iloc[index]['Dimensions'] + "x" + df.iloc[index]['Dimensions']

# We can see that some of the items do not have a specified dimension, would rather delete
# those items since there's not much we can do with badly catalogued items like these.
df = df[~df['Dimensions'].str.contains("unavailabl", na=False)]
# besides unaivalable, there are: irregular, trimmed, bookletclosed, necklace, includingfur, eachscroll,
# Left, withbase, N.A, Eachapprox., each, bottomsection, variabl, topsection, trimmedtoplate, 
# also in these forms: 511/16in., SealFace;1.12x0.74\r\nHeight:0.56\r\nStringHole..., [nodimensionsavailable


# below we look for each object's dimensions, and if it doesn't contain a digit, we remove the row
# re.search(r'\d', inputString)
for index, dimensions in enumerate(df['Dimensions']):
    string = df.iloc[index]['Dimensions']
    nr_d = 0
    for c in string:
        if c.isdigit():
            nr_d += 1
    if (nr_d == 0):
        df.at[index, 'Dimensions'] = 'DELETE'

df = df[~df['Dimensions'].str.contains("DELETE", na=False)]

df['AccessionYear'] = df['AccessionYear'].astype('float')
df['AccessionYear'] = df['AccessionYear'].astype('Int32')

df['Object ID'] = df['Object ID'].astype('Int64')
df['Dimensionality'] = df['Dimensionality'].fillna(0).astype('Int32')

df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'], errors='coerce')
df['Object End Date'] = pd.to_numeric(df['Object End Date'], errors='coerce')

df['Artist Begin Date'] = pd.to_numeric(df['Artist Begin Date'], errors='coerce')
df['Artist End Date'] = pd.to_numeric(df['Artist End Date'], errors='coerce')

for index_item, item in df.iterrows():

    if (df['Dimensionality'][index_item] == 2):
        number_dimensions = []
        text_dimension = df['Dimensions'][index_item]
        number_dimensions = [float(s) for s in re.findall(r'-?\d+\.?\d*', text_dimension)]

        covered_area = np.prod(np.array(number_dimensions))
        df['Covered Area'][index_item] = covered_area

# Unfortunately, there are a couple dimensions that couldn't have been extracted
# given a really bad input when the item was introduced into the DB by the employees
# Since part of our analysis needs the area covered by each painting/drawing, we will only keep
# the ones with a good input
df = df[df['Dimensionality'] == 2]

# Clean up of the Country column
# USA -> United States
# US -> United States
# The United States -> United States
df['Country'] = df['Country'].str.replace('USA', 'United States')
df['Country'] = df['Country'].str.replace('US', 'United States')
df['Country'] = df['Country'].str.replace('The United States', 'United States')

df['years_worked_item'] = df['Object End Date'] - df['Object Begin Date']
df['years_worked_item'] = pd.to_numeric(df['years_worked_item'], errors='coerce')

df['years_lived_artist'] = df['Artist End Date'] - df['Artist Begin Date']
df['years_lived_artist'] = pd.to_numeric(df['years_lived_artist'], errors='coerce')

# There are a couple outliers that have to be ignored throughout the plotting
# Definitely badly inserted data.
# Otherwise, a few 6000 values would show up
df = df[df['years_lived_artist'] < 120]

# Assuming we are not interesting in the extremely tiny ones
df = df[df['Covered Area'] > 10]

# There's only 1 extreme outlier that we'd like removed
df = df[df['Covered Area'] < 200000]

sns.lmplot('Object End Date', 'Covered Area', df, hue='Classification', fit_reg=False)
plt.title("Year when the painting/drawing was finished VS area covered by the item (CM^2)")
plt.xlabel("Year")
plt.ylabel("CM^2")
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

sns.lmplot('Artist Begin Date', 'years_lived_artist', df, fit_reg=False)
plt.title("Year of birth VS Years lived in total by artist")
plt.xlabel("Year of birth")
plt.ylabel("Years lived by the item's artist")
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

sns.lmplot('Artist End Date', 'Covered Area', df, hue='Culture', fit_reg=False)
plt.title("Year when the artist finished the item VS Area covered by item")
plt.xlabel("Year of finishing item")
plt.ylabel("Covered area (CM^2)")
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

sns.lmplot('years_worked_item', 'Covered Area', df, hue='Classification', fit_reg=False)
plt.title("Years worked on the item vs Covered Area (CM^2)")
plt.xlabel("Years worked")
plt.ylabel("CM^2")
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()

df["Object End Date"].plot(kind="hist")
plt.title("Distribution of the years when paintings/drawings were finished")
plt.xlabel("Year")
plt.ylabel("CM^2")
plt.show()

df['Country'].value_counts().plot(kind='bar')
plt.title("Counts for items from each country")
plt.xlabel("Country")
plt.ylabel("Number of items from the artist")
plt.show()

df['Classification'].value_counts().plot(kind='bar')
plt.title("Counts for artists from each classification of items")
plt.xlabel("Classification")
plt.ylabel("Number of items from class")
plt.show()

df['Dimensionality'].value_counts().plot(kind='bar')
plt.title("Counts for 2D and 3D items being stored")
plt.xlabel("Dimensionality")
plt.ylabel("Number of items in the dimensionality")
plt.show()

# We have to convert them back to int
df['Object Begin Date'] = df['Object Begin Date'].astype('Int32')
df['Object End Date'] = df['Object End Date'].astype('Int32')

df['Artist Begin Date'] = df['Artist Begin Date'].astype('Int32')
df['Artist End Date'] = df['Artist End Date'].astype('Int32')

# del df['years_lived_artist']
# del df['years_worked_item']


df_new = df[['Artist Display Name']]

unique_names = df_new['Artist Display Name'].unique()
unique_names = pd.DataFrame(unique_names)
unique_names['Count'] = 0
unique_names['Total Covered Area'] = 0
unique_names['Average Covered Area'] = 0

unique_names['Years Worked Average'] = 0
unique_names['Years Worked Total'] = 0

unique_names.rename(columns={0: 'Name'}, inplace=True)

unique_names['Works IDs'] = ''

for index_item, item in df.iterrows():
    for index_artist, artist in unique_names.iterrows():

        if df['Artist Display Name'][index_item] == unique_names['Name'][index_artist]:
            unique_names['Count'][index_artist] += 1
            unique_names['Total Covered Area'][index_artist] += df['Covered Area'][index_item]
            unique_names['Years Worked Total'][index_artist] += df['years_worked_item'][index_item]
            # unique_names['Works IDs'] += (df['Object ID'][index_item] + '|')

            # unique_names['Works'] = unique_names['Works'].apply(lambda x: x + [df['Object ID'][index_item]])

unique_names['Average Covered Area'] = unique_names.apply(lambda x: x['Total Covered Area'] / x['Count'], axis=1)
unique_names['Years Worked Average'] = unique_names.apply(lambda x: x['Years Worked Total'] / x['Count'], axis=1)

# keep only the artists with at least 2 paintings at Met
unique_names = unique_names[unique_names['Count'] > 1]


print(unique_names)
print("----------------------------------------------------")
print(df.head(10))
