import pickle
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
df = pd.read_csv('iris.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

classifier = RandomForestClassifier()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

new_y_test_1sample = [10,0.2,0.5,1]

score = accuracy_score(y_test,y_pred)
print(score)


pickle_out = open('model_iris.pkl', 'wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()


# # Path: knn.py
# score_test = []
# models = []

# k = np.range(1, 30)
# for i in k:
#     classifer = KNeighborsClassifier(n_neighbors=i)
#     classifer.fit(X_train, y_train)
#     score_test.append()
#     models.append(classifer)


# pikle.dump(models, open('iris_model.pkl', 'wb'))
