from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from utils import load_images
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV

# load images 
images, labels = load_images()
images = images[0:10000]
labels = labels[0:10000]
hogs = []
for image in images:
    hogs.append(hog(image, pixels_per_cell=(7, 7), cells_per_block=(4, 4)))
print("Done")
X_train, X_test, y_train, y_test = train_test_split(hogs, labels, test_size=.15)

grid_params = {
    'n_neighbors': [5, 7, 9],
    'weights': ['uniform', 'distance']
}

# Fit a KNN classifier on the training set
search = GridSearchCV(KNeighborsClassifier(), grid_params)
search.fit(X_train, y_train)
total = 0
right = 0
for image, label in zip(X_test, y_test):
    if search.predict(image.reshape(1, -1)) == label:
        right += 1
    total += 1

print(right / total)
