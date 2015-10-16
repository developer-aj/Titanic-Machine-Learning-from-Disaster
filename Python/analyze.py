import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import mode

def clean_data(data):
    data.Fare = data.Fare.map(lambda x: np.nan if x==0 else x)
    classmeans = data.pivot_table('Fare', columns='Pclass', aggfunc='mean')
    data.Fare = data[['Fare','Pclass']].apply(lambda x: classmeans[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)

    meanAge = np.mean(data.Age)
    data.Age = data.Age.fillna(meanAge)

    data.Cabin = data.Cabin.fillna('Unknown')

    modeEmbarked = mode(data.Embarked)[0][0]
    data.Embarked = data.Embarked.fillna(modeEmbarked)

    return data


def plot_data(data):
    def proportionSurvived(discreteVar):
        by_var = data.groupby([discreteVar, 'Survived'])

        table = by_var.size()
        print table
        table = table.unstack()
        normedtable = table.div(table.sum(1), axis=0)
        return normedtable

    discreteVarList = ['Sex', 'Pclass', 'Embarked']
    fig1, axes1 = plt.subplots(3, 1)

    for i in range(3):
        var = discreteVarList[i]
        table = proportionSurvived(var)
        table.plot(kind='barh', stacked=True, ax=axes1[i])
    fig1.show()

def main():
    path = '../Data/'

    print 'Loading data...'
    train_df = pd.read_csv(path + 'train.csv')
    test_df = pd.read_csv(path + 'test.csv')

    print 'Cleaning data...'
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print 'Plotting data...'
    plot_data(train_df)

if __name__ == '__main__':
    main()
