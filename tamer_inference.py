import cv2
import numpy as np
import torch
from torchvision import transforms

from tamer.datamodule.transforms import ScaleToLimitRange
from tamer.datamodule.vocab import vocab
from tamer.lit_tamer import LitTAMER

# Initialize vocabulary
vocab.init("data/hme100k/dictionary.txt")

# Set up device (prefer CUDA if available, fallback to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model checkpoint (hme100k, without fusion)
inference_checkpoint = (
    "lightning_logs/version_31/checkpoints/epoch=4-step=125.ckpt"
)
inference_model = LitTAMER.load_from_checkpoint(inference_checkpoint)
inference_model.eval()

# Path to the input inference image
inference_image_path = "proba_1.png"

# Load image using OpenCV
inference_image = cv2.imread(inference_image_path)

# Convert BGR (OpenCV default) to RGB
inference_image_rgb = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB)

# Convert RGB image to grayscale
inference_image_gray = cv2.cvtColor(inference_image_rgb, cv2.COLOR_RGB2GRAY)

# Size limits for scaling transformations
K_MIN = 0.7
K_MAX = 1.4
H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024

# Compose transforms: PIL conversion, grayscale, to tensor normalized [0,1]
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

# Apply transformations to grayscale image
inference_image_tensor = transform(inference_image_gray)
# print("[Script] Tensor shape:", inference_image_tensor.shape)
# print("[Script] Tensor dtype:", inference_image_tensor.dtype)
# print("[Script] Tensor min/max:", inference_image_tensor.min(), inference_image_tensor.max())

# Add batch dimension
inference_image_input = inference_image_tensor.unsqueeze(dim=0)

# Scale image tensor to match training data size limits
scaler = ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI)
inference_image_input_np = inference_image_input.squeeze(0).squeeze(0).numpy()
scaled_inference_image_np = scaler(inference_image_input_np)

scaled_inference_image_tensor = torch.from_numpy(scaled_inference_image_np).unsqueeze(0).unsqueeze(0)
# print("[Script] Scaled tensor shape:", scaled_inference_image_tensor.shape)
# print("[Script] Scaled tensor dtype:", scaled_inference_image_tensor.dtype)
# print("[Script] Scaled tensor min/max:", scaled_inference_image_tensor.min(), scaled_inference_image_tensor.max())

# Prepare input tensor for inference
inference_image_input = scaled_inference_image_tensor

# Create a boolean mask tensor of zeros with the same spatial size as input image
h, w = inference_image_input.shape[2], inference_image_input.shape[3]
inference_mask = torch.zeros(1, h, w, dtype=torch.bool)

# Move model and tensors to the device (GPU or CPU)
inference_model = inference_model.to(device)
inference_image_input = inference_image_input.to(device)
inference_mask = inference_mask.to(device)

# Perform inference without gradient calculations
with torch.no_grad():
    inference_hyp = inference_model.approximate_joint_search(inference_image_input, inference_mask)


# inference_hyp is a Hypothesis object of length 1 containing the score and seq of best prediction

# Extract scores for each hypothesis and get the best scoring one
score = inference_hyp[0].score

# Convert indices sequence to words using vocabulary
prediction = vocab.indices2words(inference_hyp[0].seq)

print(f"Prediction: {prediction}")
probability = np.exp(score)
print(f"Probability: {probability}")


# Currently when the model is loaded warnings are present because of the discrepancy of lightning
# version numbers. Will train the model on crohme and hme100k and other datasets in the future with
# the newest versions of everything, until then it is like this.
