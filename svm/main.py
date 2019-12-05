from sklearn import svm
import sklearn.model_selection as sk

from utils import load_images

images, labels = load_images()

x_train, x_test, y_train, y_test = sk.train_test_split(images, labels, test_size=.15)

model = svm.SVC()
model.fit(x_train, y_train)
predictions = [int(pred) for pred in model.predict(x_test)]
correct = sum(int(pred == y) for pred, y in zip(predictions, y_test))
print(100*correct/len(predictions))
