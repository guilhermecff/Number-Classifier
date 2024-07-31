import io
import pickle
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load the pre-trained model
with open('mnist_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = FastAPI()

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # Read the image file
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert('L')
        
        # Invert the image colors
        image = ImageOps.invert(image)
        
        # Resize the image to 28x28
        image = image.resize((28, 28), Image.LANCZOS)
        
        # Convert image to numpy array and reshape for the model
        img_array = np.array(image).reshape(1, -1)
        
        # Predict using the pre-trained model
        prediction = model.predict(img_array)
        
        return {'prediction': int(prediction[0])}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
