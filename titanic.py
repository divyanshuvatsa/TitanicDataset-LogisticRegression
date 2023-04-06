import pandas as pd

training_dataset = pd.read_csv("0000000000002429_training_titanic_x_y_train.csv")
testing_dataset = pd.read_csv("0000000000002429_test_titanic_x_test.csv")

print(training_dataset.isnull().sum())

print(testing_dataset.isnull().sum())

from sklearn.impute import SimpleImputer 
import numpy as np

training_dataset

training_dataset = training_dataset.drop(['Cabin'], axis = 1)

testing_dataset = testing_dataset.drop(['Cabin'], axis = 1)

training_dataset = training_dataset.drop(["Name","Ticket"], axis = 1)

testing_dataset = testing_dataset.drop(["Name","Ticket"],axis = 1)

training_dataset.head()

testing_dataset.head()

gen = pd.get_dummies(training_dataset['Sex'])
gen

gen2 = pd.get_dummies(testing_dataset['Sex'])
gen2

training_dataset_new = pd.concat([training_dataset,gen],axis=1)
training_dataset_new

testing_dataset_new = pd.concat([testing_dataset,gen2],axis=1)
testing_dataset_new

training_dataset_new.drop(['Sex','female'], inplace=True,axis=1)

testing_dataset_new.drop(['Sex','female'], inplace=True,axis=1)

testing_dataset_new

gen3 = pd.get_dummies(training_dataset_new['Embarked'])
gen3

gen4 = pd.get_dummies(testing_dataset_new['Embarked'])
gen4

train_dataset_final = pd.concat([training_dataset_new,gen3],axis=1)
train_dataset_final

test_dataset_final = pd.concat([testing_dataset_new,gen4],axis=1)
test_dataset_final

train_dataset_final.drop(['Embarked','C'], inplace=True,axis=1)

test_dataset_final.drop(['Embarked','C'], inplace=True,axis=1)

train_dataset_final

test_dataset_final

train_dataset_final.isnull().sum()

test_dataset_final.isnull().sum()

type(train_dataset_final)

train = train_dataset_final.iloc[:,:].values
train

test = test_dataset_final.iloc[:,:].values
test

train[:,1:3].shape

s_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
s_imputer = s_imputer.fit(train[:,1:3])
train[:,1:3] = s_imputer.transform(train[:,1:3])

s_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
s_imputer = s_imputer.fit(test[:,1:3])
test[:,1:3] = s_imputer.transform(test[:,1:3])

train[0]

test[0]

train = pd.DataFrame(train)
train

test = pd.DataFrame(test)
test

print(train.isnull().sum())

print(test.isnull().sum())

x_test = test.iloc[:,:].values
x_test[0]

y_train = train.iloc[:,5].values 
y_train

part_1 = train.iloc[:,:5].values
part_1[0]

part_2 = train.iloc[:,6:].values
part_2[0]

x_train = np.hstack((part_1,part_2))
x_train[0]

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train

x_test

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=10)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)
y_pred

np.savetxt('logreg_cn.csv', y_pred)
