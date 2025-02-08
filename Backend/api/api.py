from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from my_model import DeepfakeModel  # Import your trained model
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = DeepfakeModel()  # Make sure this class is properly defined in my_model.py
model.load_state_dict(torch.load("deepfake_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Define image preprocessing function
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match model input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize (update if needed)
    ])
    return transform(image)

# API Endpoint for Prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")  # Ensure it's RGB
    image_tensor = preprocess(image)  # Preprocess image
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        prediction = model(image_tensor)  # Get model output

    return {"is_fake": prediction.item() > 0.5}  # Adjust threshold as needed

# Run FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
