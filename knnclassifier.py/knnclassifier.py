#!pip install --upgrade tensorflow

import numpy as np
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from utils import load_images
# load images 
images, labels = load_images()

X_train, X_test, y_train, y_test = sk.train_test_split(images, labels, test_size=.15)


X = images 
y = labels

# Apply PCA by fitting the data with only 60 dimensions
pca = PCA(n_components=60).fit(X)
# Transform the data using the PCA fit above
X = pca.transform(X)
y = y.values

# Shuffle and split the dataset into the number of training and testing points 
model = train_test_split.StratifiedShuffleSplit(y, 3, test_size=0.4, random_state=42)
for train_index, test_index in model:
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

# Fit a KNN classifier on the training set
knn_clf = KNeighborsClassifier(n_neighbors=3, p=2)
knn_clf.fit(X_train, y_train)

# Initialize the array of predicted labels
y_pred = np.empty(len(y_test), dtype=np.int)

start = time()

# Find the nearest neighbors indices for each sample in the test set
kneighbors = knn_clf.kneighbors(X_test, return_distance=False)

# For each set of neighbors indices
for idx, indices in enumerate(kneighbors):
  # Find the actual training samples & their labels
  neighbors = [X_train[i] for i in indices]
  neighbors_labels = [y_train[i] for i in indices]
  
svm_clf = svm.SVC(C=0.5, kernel='rbf', decision_function_shape='ovo', random_state=42)
svm_clf.fit(neighbors, neighbors_labels)
label = svm_clf.predict(X_test[idx].reshape(1, -1))
y_pred[idx] = label


#print(accuracy_score(y_test, y_pred))