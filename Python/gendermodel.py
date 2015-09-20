import numpy as np
import csv as csv

#training set
#open up the file
csv_file = open('../Data/train.csv','rb')
csv_obj = csv.reader(csv_file)
train_header = csv_obj.next()

data = []
for row in csv_obj:
    data.append(row)
data = np.array(data)

number_passengers = np.size(data[0::,1])
number_survived = np.sum(data[0::,1].astype(np.float))

women = data[0::,4] == "female"
men = data[0::,4] != "female"

women_onboard = data[women, 1].astype(np.float)
men_onboard = data[men, 1].astype(np.float)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived
csv_file.close()

#test set
test_file = open('../Data/test.csv', 'rb')
test_obj = csv.reader(test_file)
test_header = test_obj.next()

prediction_file = open('genderbasedmodel.csv', 'wb')
prediction_obj = csv.writer(prediction_file)

prediction_obj.writerow(['PassengerId', 'Survived'])
for row in test_obj:
    if row[3] == 'female':
        prediction_obj.writerow([row[0], '1']) #Survived
    else:
        prediction_obj.writerow([row[0], '0']) #Not Survived
test_file.close()
prediction_file.close()
