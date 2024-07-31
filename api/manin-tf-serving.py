from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()


# MODEL = tf.saved_model.load('C:/Users/nites/OneDrive/Desktop/DL_Projects/potato_diseas_prediction/saved_models/1/content/1')
MODEL = tf.keras.models.load_model('../saved_models/potato_disease_prediction.keras')
# MODEL = tf.keras.layers.TFSMLayer('../saved_models/1', call_endpoint='serve',call_endpoint='serving_default')
# MODEL = tf.keras.models.load_model('.../saved_models/1')
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get('/ping')
async def ping():
    return "hello i am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(image_batch)

    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    print(predicted_class, confidence)
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }
    pass


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
