import numpy as np
from load_and_process import load_fer2013
from load_and_process import preprocess_input

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

batch_size = 32
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.3,shuffle=True)

# emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
# print(data_generator.flow(xtest))
# predicted = []
#
# for test_data in xtest:
#     preds = emotion_classifier.predict(test_data)[0]
#     emotion_probability = np.max(preds)
#     predicted.append(EMOTIONS[preds.argmax()])

# print(ytest)
validation_generator = data_generator.flow(xtest,ytest,

                                                        batch_size=batch_size,
                                                        )


predicted = emotion_classifier.predict_generator(validation_generator)
predicted = np.argmax(predicted,axis = 1)

ytest = np.argmax(validation_generator.y,axis=1)
# predicted = emotion_classifier.predict(xtest)

print(accuracy_score(ytest,predicted))
print("Confusion")
confusion_matrix =  confusion_matrix(ytest,predicted)
confusion_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,display_labels=EMOTIONS)
print(confusion_disp)
confusion_disp.plot()
plt.show()