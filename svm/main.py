from skimage.feature import hog
from sklearn import svm
import sklearn.model_selection as sk
from sklearn.decomposition import PCA
from utils import load_images

images, labels = load_images()
images = images[0:10000]
labels = labels[0:10000]
hogs = []
for image in images:
    hogs.append(hog(image, pixels_per_cell=(7, 7), cells_per_block=(4, 4)))
print("Done")
X_train, X_test, y_train, y_test = sk.train_test_split(hogs, labels, test_size=.15)
pca = PCA(n_components=60)
X_train = pca.fit_transform(X_train)
# Transform the data using the PCA fit above
X_test = pca.transform(X_test)

model = svm.SVC()
model.fit(X_train, y_train)
predictions = [int(pred) for pred in model.predict(X_test)]
correct = sum(int(pred == y) for pred, y in zip(predictions, y_test))
print(100*correct/len(predictions))
