import pandas as pd
import numpy as np
import pylab as P

df = pd.read_csv('../Data/train.csv', header=0)

# Adding new column 'Gender'
df['Gender'] = df.Sex.map({'female': 0, 'male': 1}).astype(int)

# Adding new column 'EmbarkedValues'
df['EmbarkedValues'] = df.Embarked.dropna().map({'C': 1, 'Q': 2, 'S': 3}).astype(int)
df.loc[(df.Embarked.isnull()), 'EmbarkedValues'] = 0
#print df[df.Embarked.isnull()][['Embarked', 'EmbarkedValues']]

# Calculating median based on Gender and Pclass
median_ages = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i, j] = df[(df.Gender == i) & (df.Pclass == j+1)]['Age'].dropna().median()

# Adding new column 'AgeFill'
df['AgeFill'] = df['Age']

# Filling NA values with median values
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i, j]

# Adding new column 'AgeIsNull'
df['AgeIsNull'] = pd.isnull(df['Age']).astype(int)

# Adding new column 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch']

# 'Age*Pclass'
df['Age*Pclass'] = df['AgeFill'] * df['Pclass']

# dropping columns
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

#converting to numpy array
train_data = df.values

print df.head(10)
print train_data

print 'Drawing histogram...'
df['Age*Pclass'].hist(bins=16, range=(0,80), alpha=0.5)
P.show()
print 'Done...'
