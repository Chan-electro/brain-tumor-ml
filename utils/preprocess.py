import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def load_and_preprocess_image(file_storage):
    """
    file_storage is the uploaded file from Flask (werkzeug FileStorage).
    Returns image tensor ready for model: shape (1, 224, 224, 3)
    """
    image = Image.open(file_storage).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image
