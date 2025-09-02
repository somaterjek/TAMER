from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import torch
import cv2
from torchvision import transforms

from tamer.datamodule.transforms import ScaleToLimitRange
from tamer.datamodule.vocab import vocab
from tamer.lit_tamer import LitTAMER

# Initialize model and vocab once at startup
vocab.init("data/hme100k/dictionary.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inference_checkpoint = "lightning_logs/version_1/checkpoints/epoch=51-step=162967-val_ExpRate=0.6851.ckpt"
inference_model = LitTAMER.load_from_checkpoint(inference_checkpoint)
inference_model.eval()
inference_model = inference_model.to(device)

# Constants
K_MIN = 0.7
K_MAX = 1.4
H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

def run_inference(image_np):
    # Preprocessing (adapted from your script)
    image_tensor = transform(image_np)
    image_input = image_tensor.unsqueeze(dim=0)
    scaler = ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)
    image_input_np = image_input.squeeze(0).squeeze(0).numpy()
    scaled_image_np = scaler(image_input_np)
    scaled_image_tensor = torch.from_numpy(scaled_image_np).unsqueeze(0).unsqueeze(0)
    image_input = scaled_image_tensor
    h, w = image_input.shape[2], image_input.shape[3]
    inference_mask = torch.zeros(1, h, w, dtype=torch.bool)
    image_input = image_input.to(device)
    inference_mask = inference_mask.to(device)
    with torch.no_grad():
        inference_hyp = inference_model.approximate_joint_search(image_input, inference_mask)
    score = inference_hyp[0].score
    prediction = vocab.indices2words(inference_hyp[0].seq)
    return prediction, float(score)

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)
    prediction, score = run_inference(image_np)
    return {"latex": prediction, "score": score}