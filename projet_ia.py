from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np

model = InceptionV3(weights='imagenet')

def preprocess_image(img_path):
    imgs = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 prend en charge 299x299 images
    img_array = image.img_to_array(imgs)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

image_path = 'doggy.jpg'
img = preprocess_image(image_path)

predictions = model.predict(img)

decoded_predictions = decode_predictions(predictions, top=3)[0]
print("Prédictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
