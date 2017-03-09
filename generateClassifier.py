# Import the modules
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

# Load the dataset
print("Loading MNIST Dataset")
dataset = datasets.fetch_mldata("MNIST Original")
print("Finished loading MNIST Dataset")

# Extract the features and labels
features = np.array(dataset.data, np.int16)
labels = np.array(dataset.target, int)

# Extract the hog features
list_hog_fd = []
list_labels = []
for i, feature in enumerate(features):
    if labels[i] == 0:
        continue
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
    list_labels.append(labels[i])
hog_features = np.array(list_hog_fd, np.float64)
labels = np.array(list_labels, int)

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump(clf, "digits_cls.pkl", compress=3)
print("Finished generating classifier")
