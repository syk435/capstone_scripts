from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import sys
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from utils import data_processing
from utilsimport get_labels
from utils import detect_faces
from utils import draw_text
from utils import draw_bounding_box
from utils import apply_offsets
from utils import load_detection_model
from utils import preprocess_input
sys.path.append('../classifier-metrics')
from metrics.utils import generate_classification_report, generate_confusion_matrix
import numpy as np


# create the resenet model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights(sys.argv[1])
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')

# create the vgg16 model
base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x1 = base_model1.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dense(1024, activation='relu')(x1)
predictions1 = Dense(1, activation='sigmoid')(x1)
model1 = Model(inputs=base_model1.input, outputs=predictions1)
model1.load_weights(sys.argv[2])
model1.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')

# create the inceptionv3 model
base_model2 = InceptionV3(weights='imagenet', include_top=False)
x2 = base_model2.output
x2 = GlobalAveragePooling2D()(x2)
x2 = Dense(1024, activation='relu')(x2)
predictions2 = Dense(1, activation='sigmoid')(x2)
model2 = Model(inputs=base_model2.input, outputs=predictions2)
model2.load_weights(sys.argv[3])
model2.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')

images, true_labels = data_processing.read_images_and_labels(test_dict, int(sys.argv[4]), int(sys.argv[5]))
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
emotion_window = []
for image in images
	faces = face_cascade.detectMultiScale(images, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	for face_coordinates in faces:
		emotion_offsets = (20, 40)
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
		#img = load_img(sys.argv[1],False,target_size=(int(sys.argv[2]),int(sys.argv[3])))
		#x = img_to_array(img)
		#x = np.expand_dims(x, axis=0)
		raw_predictions = model.predict(gray_face)
		#preds = np.rint(raw_predictions)
		preds = raw_predictions
		preds1 = model1.predict(gray_face)
		preds2 = model2.predict(gray_face)
		final_preds = []
		for p in range(0,len(preds)):
		    av = (preds[p][0]+preds1[p][0]+preds2[p][0])/3.0
		    final_preds.append([av])
	finals = np.rint(final_preds)