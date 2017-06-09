from sklearn.tree import DecisionTreeClassifier
import pandas as pd

train = pd.read_csv('/Users/namakilam/workspace/indiahacks_2017/predictRoadSign/dataset/train.csv')
test =  pd.read_csv('/Users/namakilam/workspace/indiahacks_2017/predictRoadSign/dataset/test.csv')
train.head()
test.head()
mapping = {'Front' : 0, 'Left' : 1, 'Rear' : 2, 'Right' : 3}
train.rename(columns = {'SignFacing (Target)' : 'Target'}, inplace=True)
train = train.replace({ 'DetectedCamera' : mapping , 'Target' : mapping})
test = test.replace( { 'DetectedCamera' : mapping} )


y_train = train['Target']
test_Id = test['Id']


train.drop(['Id', 'Target'], inplace = True, axis = 1)
test.drop(['Id'], inplace = True, axis = 1)


clf = DecisionTreeClassifier(max_depth = 6, max_features = 4, random_state = 0)
clf.fit(train, y_train)
pred = clf.predict_proba(test)

outputColumns = ['Front','Left','Rear','Right']
res = pd.DataFrame(data=pred, columns=outputColumns)
res['Id'] = test_Id
res = res[['Id','Front','Left','Rear','Right']]
res.to_csv("dt_classifier3.csv", index=False)
