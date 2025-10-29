import joblib
import torch
from torch import nn
import os

current_dir = os.path.dirname(__file__)
def load_preprocessors():
    scaler = joblib.load(os.path.join(current_dir, "model", "scaler.pkl"))
    encoder = joblib.load(os.path.join(current_dir, "model", "encoder.pkl"))
    global_means = joblib.load(os.path.join(current_dir, "model", "global_means.pkl"))
    return scaler, encoder, global_means

scaler, encoder, global_means = load_preprocessors()

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    def forward(self, x):
        return self.encoder(x)
    
def load_autoencoder() -> AutoEncoder:
    # 加载 encoder文件
    input_dim = 436
    ae = AutoEncoder(input_dim=input_dim, latent_dim=128)
    ae.encoder.load_state_dict(torch.load(os.path.join(current_dir, "model", "encoder_model.pt"), map_location="cpu"))
    ae.eval()
    return ae

autoencoder = load_autoencoder()

def autoencoder_user_req_emb(user_vec):
    with torch.no_grad():
        _, user_emb = autoencoder(torch.tensor(user_vec, dtype=torch.float32).unsqueeze(0))

        user_emb_np = user_emb.numpy()[0].tolist()
        print("用户 embedding 向量维度:", len(user_emb_np))
        return user_emb_np