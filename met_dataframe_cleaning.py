import matplotlib
from tabulate import tabulate
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\Tudor\PycharmProjects\Intelligent\MetObjects.txt", dtype={
    'Gallery Number': 'object', 'Period': 'object',
    'Portfolio': 'object', 'AccessionYear': 'object',
    'River': 'object', 'Subregion': 'object',
    'Dynasty': 'object', 'Excavation': 'object',
    'Locale': 'object', 'Locus': 'object', 'Reign': 'object'
}, low_memory=False)

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

pd.set_option('display.max_columns', None)

# print(tabulate(df, headers = 'keys', tablefmt = 'psql'))

print(df.dtypes)

df['Department'].value_counts().plot(kind='bar')
plt.title("Counts for items in each one of the departments")
plt.xlabel("Department")
plt.ylabel("Number of stored items")
plt.show()

df = df[df['Object Number'].notna()]

# Issue #1: The object number is a string stored like it would be some sort of IP address
# df['Object Number'] = (df['Object Number'].str.split()).apply(lambda x: (x[0].replace('.', '')))

# There is an another column <Object ID> that has the same scope as <Object Number>
del df['Object Number']
# Re-order the columns such that <Object ID> is the first column.
# Other reorders should follow
df = df[['Object ID', 'Is Highlight', 'Is Timeline Work', 'Department', 'Object Name', 'Title', 'Culture', 'Period',
         'Dynasty', 'Reign', 'Is Public Domain', 'Gallery Number', 'AccessionYear',
         'Portfolio', 'Constiuent ID', 'Artist Role', 'Artist Prefix', 'Artist Display Name', 'Artist Display Bio',
         'Artist Suffix', 'Artist Alpha Sort', 'Artist Nationality', 'Artist Begin Date',
         'Artist End Date', 'Artist Gender', 'Artist ULAN URL', 'Artist Wikidata URL', 'Object Date',
         'Object Begin Date', 'Object End Date', 'Medium', 'Dimensions', 'Credit Line',
         'Geography Type', 'City', 'State', 'Country', 'Region', 'Subregion', 'Locale', 'Locus', 'Excavation', 'River',
         'Classification', 'Rights and Reproduction', 'Link Resource',
         'Object Wikidata URL', 'Metadata Date', 'Repository', 'Tags', 'Tags AAT URL', 'Tags Wikidata URL']]

print(df.head(10))
