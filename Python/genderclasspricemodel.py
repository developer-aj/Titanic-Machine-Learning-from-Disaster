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

women = data[0::,4] == "female"
men = data[0::,4] != "female"

# So we add ceiling
fare_ceiling = 40
#then modify the data in the fare column to = 39, if it is greater or equal to the ceiling
data[data[0::, 9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size

#I know there were 1st, 2nd and 3rd classes on board
number_of_classes = 3

#But it's better practice to calculate this from the data directly
#Take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::, 2]))

#initialize the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets), float)

for i in xrange(number_of_classes): #loop through each class
    for j in xrange(number_of_price_brackets):  #loop through each price bin

        women = data[
                (data[0::, 4] == 'female')      #is a female
                &(data[0::, 2].astype(np.float) == i+1)     #was ith class
                &(data[0:, 9].astype(np.float) >= j*fare_bracket_size) #was greater than this bin
                &(data[0:, 9].astype(np.float) < (j+1)*fare_bracket_size)  #and less than the next bin in the 2nd col
                ,1]

        men = data[
                (data[0::, 4] != 'female')
                &(data[0::, 2].astype(np.float) == i+1)
                &(data[0:, 9].astype(np.float) >= j*fare_bracket_size)
                &(data[0:, 9].astype(np.float) < (j+1)*fare_bracket_size)
                ,1]

        survival_table[0,i,j] = np.mean(women.astype(np.float))  # Female stats
        survival_table[1,i,j] = np.mean(men.astype(np.float))    # Male stats

# Since in python if it tries to find the mean of an array with nothing in it
# (such that the denominator is 0), then it returns nan, we can convert these to 0
# by just saying where does the array not equal the array, and set these to 0.
survival_table[ survival_table != survival_table ] = 0.

# Now I have my proportion of survivors, simply round them such that if <0.5
# I predict they dont surivive, and if >= 0.5 they do
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1
csv_file.close()
print survival_table

#test
test_file = open('../Data/test.csv', 'rb')
test_obj = csv.reader(test_file)
header = test_obj.next()

predictions_file = open('genderclassmodel.csv', 'wb')
predictions_obj = csv.writer(predictions_file)
predictions_obj.writerow(['PassengerId', 'Survived'])

#First thing to do is bin up the price file
for row in test_obj:
    for j in xrange(number_of_price_brackets):
        try:
            row[8] = float(row[8])

        except:
            bin_fare = 3 - float(row[1])
            break
        
        if row[8] > fare_ceiling:
            bin_fare = number_of_price_brackets - 1
            break

        if row[8] >= j*fare_bracket_size and row[8] < (j+1)*fare_bracket_size:
            bin_fare = j
            break

    if row[3] == 'female':
        predictions_obj.writerow([row[0], '%d' % int(survival_table[0, float(row[1]) - 1, bin_fare ])])
    else:
        predictions_obj.writerow([row[0], '%d' % int(survival_table[1, float(row[1]) - 1, bin_fare ])])

# Close out the files
test_file.close()
predictions_file.close()
