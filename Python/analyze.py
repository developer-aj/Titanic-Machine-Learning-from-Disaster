import pandas as pd
import numpy as np
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

def main():
    path = '../Data/'
    train_df = pd.read_csv(path + 'train.csv')

    # Description of data
    print train_df.describe()
    train_df = clean_data(train_df)
    print train_df

if __name__ == '__main__':
    main()
