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
        table = table.unstack()
        normedtable = table.div(table.sum(1), axis=0)
        return normedtable

    discreteVarList = ['Sex', 'Pclass', 'Embarked']
    fig1, axes1 = plt.subplots(3, 1)

    for i in range(3):
        var = discreteVarList[i]
        table = proportionSurvived(var)
        table.plot(kind='barh', stacked=True, ax=axes1[i])
    #fig1.show()

    # Plot based on categories
    fig2, axes2 = plt.subplots(2, 3)
    genders = data.Sex.unique()
    classes = data.Pclass.unique()
    def normrgb(rgb):
        rgb = [float(x)/255 for x in rgb]
        return rgb
    
    darkpink, lightpink = normrgb([255, 20, 147]), normrgb([255, 182, 193])
    darkblue, lightblue = normrgb([0, 0, 128]), normrgb([135, 206, 250])
    for gender in genders:
        for pclass in classes:
            if gender == 'male':
                colorscheme = [lightblue, darkblue]
                row = 0
            else:
                colorscheme = [lightpink, darkpink]
                row = 1
            group = data[(data.Sex==gender)&(data.Pclass==pclass)]
            group = group.groupby(['Embarked', 'Survived']).size().unstack()
            group = group.div(group.sum(1), axis=0)
            group.plot(kind='barh', ax=axes2[row, (int(pclass)-1)], color=colorscheme, stacked=True, legend=False).set_title('Class '+str(pclass)).axes.get_xaxis().set_ticks([])

    plt.subplots_adjust(wspace=0.4, hspace=1.3)
    fhandles, flabels = axes2[1,2].get_legend_handles_labels()
    mhandles, mlabels = axes2[0,2].get_legend_handles_labels()
    plt.figlegend(fhandles, ('die', 'live'), title='Female', loc='center', bbox_to_anchor=(0.06, 0.45, 1.1, .102))
    plt.figlegend(mhandles, ('die', 'live'), 'center', title='Male',bbox_to_anchor=(-0.15, 0.45, 1.1, .102))
    #fig2.show()

    # Plots using bins
    bins = [0, 5, 14, 25, 40, 60, 100]
    binNames = ['Young Child', 'Child', 'Young Adult', 'Adult', 'Middle Aged', 'Older']
    binAge = pd.cut(data.Age, bins, labels=binNames)
    binFare = pd.qcut(data.Fare, 3, labels=['Cheap', 'Middle', 'Expensive'])

    fig3, axes3 = plt.subplots(1, 2)
    binVars = [binAge, binFare]
    varNames = ['Age', 'Fare']
    badStringList = ['(', ')', 'female', 'male', ',']

    def removeBadStringFromString(string):
        for badString in ['(', ')', 'female', 'male']:
            string = string.replace(badString, '')
        return string
    
    def removeBadStringFromLabels(ax, badStringList):
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels = [removeBadStringFromString(label) for label in labels]
        return labels

    for i in range(2):
        group = data.groupby([binVars[i], 'Sex', 'Survived'])
        group = group.size().unstack()
        group = group.div(group.sum(1), axis=0)
        cols = [[lightpink, lightblue],[darkpink, darkblue]]
        group.plot(kind='barh', stacked=True, ax=axes3[i], legend=False, color=cols)
        labels = removeBadStringFromLabels(axes3[i], badStringList)
        axes3[i].set_yticklabels(labels)
        axes3[i].get_xaxis().set_ticks([])
        axes3[i].set_ylabel('')
        axes3[i].set_title(varNames[i])

        if i==1:
            axes3[i].yaxis.tick_right()
            axes3[i].yaxis.set_label_position("right")

    handles, labels = axes3[0].get_legend_handles_labels()
    plt.figlegend(handles[0], ['die', 'die'], loc='upper center')
    plt.figlegend(handles[1], ['live', 'live'], loc='lower center')

    fig3.show()

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
