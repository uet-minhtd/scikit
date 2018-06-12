import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,datasets
iris = datasets.load_iris()
#print(iris['data'])
#print(len(iris['data']))
#print(iris['target_names'])
#print "Label"
#print(iris['target'])
#print(len(iris['target']))
iris_X = iris.data
iris_y = iris.target
#print 'Number of classes :%d ' %len(np.unique(iris_y))


#X0 = iris_X[iris_y == 0,:]
#print '\nSamples from class 0:\n', X0[:5,:]

#X1 = iris_X[iris_y == 1,:]
#print '\nSamples from class 1:\n', X1[:5,:]

#X2 = iris_X[iris_y == 2,:]
#print '\nSamples from class 2:\n', X2[:5,:]

#selection model : data train : 130 item,test data train :20
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=20)

label_list = iris_y.tolist()
print "Training size: %d" %len(y_train)
list = X_train.tolist()
flower_name = ''
for i in range(len(list)):
	if label_list[i] == 0:
		flower_name = 'Iris setosa'
	elif label_list[i] == 1:
		flower_name = 'Iris virginica'
	else :
		flower_name = 'Iris versicolor'
	print str(list[i]) + " ==> "+ str(label_list[i]) + " ==> " +flower_name

	
#Bat dau day hoc cho no
#Lua chon so hang xom can lay y kien
k = 3;
clf = neighbors.KNeighborsClassifier(k, p = 2,weights = 'distance')
clf.fit(X_train, y_train)

#Ket qua duoc du doan cho tap du lieu X_test
y_pred = clf.predict(X_test)
print(y_pred)


#Kiem tra do dung dan
print "Print results for 20 test data points:"
print "Predicted labels: ", y_pred[0:19]
print "Ground truth    : ", y_test[0:19]


#Phuong phap danh gia ket qua du doan cua KNN
from sklearn.metrics import accuracy_score
print "Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred))