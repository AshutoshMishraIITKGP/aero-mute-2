import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
import streamlit as st

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="AERO-MUTE Dashboard", page_icon="🚁", layout="wide")

st.markdown("""
<style>
    /* Military Tactical Theme */
    .stApp {
        background-color: #0a0f0a;
        background-image: radial-gradient(circle at 50% 0%, #152415 0%, #0a0f0a 80%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #4ade80 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 0 10px rgba(74,222,128,0.2);
    }
    
    /* Upload Box Styling */
    .stFileUploader > div > div {
        background-color: rgba(20, 30, 20, 0.6) !important;
        border: 1px dashed #4ade80 !important;
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2e5c3a 0%, #1a3a22 100%) !important;
        border: 1px solid #4ade80 !important;
        color: #fff !important;
        box-shadow: 0 0 10px rgba(74, 222, 128, 0.2);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
        transform: translateY(-2px);
    }
    
    /* Audio Player */
    audio {
        width: 100%;
        border-radius: 5px;
        filter: sepia(20%) hue-rotate(80deg) saturate(150%) brightness(0.8);
    }
    
    /* Containers */
    .css-1r6slb0 {
        background: rgba(15, 20, 15, 0.8);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2a3a2a;
    }
</style>
""", unsafe_allow_html=True)


# 2. Architecture Definition
class FullSubNetPlus(nn.Module):
    def __init__(self, n_fft=320, hop_length=160, hidden_size=512):
        super(FullSubNetPlus, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freqs = n_fft // 2 + 1
        
        self.full_band = nn.LSTM(self.num_freqs, hidden_size, num_layers=2, batch_first=True)
        self.fb_linear = nn.Linear(hidden_size, self.num_freqs)
        
        self.sub_band = nn.LSTM(2, 384, num_layers=2, batch_first=True)
        self.sb_linear = nn.Linear(384, 2) 

    def forward(self, noisy_wave):
        noisy_wave = noisy_wave.squeeze(1)
        window = torch.hann_window(self.n_fft).to(noisy_wave.device)
        stft = torch.stft(noisy_wave, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=window, return_complex=True)
        
        noisy_mag = torch.abs(stft)
        B, F, T = noisy_mag.shape
        x = noisy_mag.permute(0, 2, 1)
        
        fb_out, _ = self.full_band(x)
        fb_out = self.fb_linear(fb_out)
        
        x_sb = x.unsqueeze(-1)
        fb_out_sb = fb_out.unsqueeze(-1)
        
        sb_input = torch.cat([x_sb, fb_out_sb], dim=-1)
        sb_input = sb_input.permute(0, 2, 1, 3).reshape(B * F, T, 2)
        
        sb_out, _ = self.sub_band(sb_input)
        mask_real_imag = self.sb_linear(sb_out)
        
        mask_real_imag = mask_real_imag.reshape(B, F, T, 2)
        mask_complex = torch.complex(mask_real_imag[..., 0], mask_real_imag[..., 1])
        
        enhanced_stft = stft * mask_complex
        enhanced_wave = torch.istft(enhanced_stft, n_fft=self.n_fft, hop_length=self.hop_length, 
                                    window=window, length=noisy_wave.shape[-1])
        return enhanced_wave.unsqueeze(1)


# 3. Model Loading (Cached to save RAM/Time on Streamlit Cloud)
@st.cache_resource
def load_model():
    device = torch.device('cpu') # Streamlit Cloud uses CPU
    model = FullSubNetPlus().to(device)
    checkpoint = torch.load("aeromute_fullsubnet_epoch_25.pt", map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, device

model, device = load_model()


# 4. Inference Function
def process_audio(audio_bytes):
    # Save uploaded bytes to a temp file
    with open("temp_in.wav", "wb") as f:
        f.write(audio_bytes)
        
    # Load and resample to 16kHz
    audio, sr = librosa.load("temp_in.wav", sr=16000, mono=True)
    
    # Cap at 10 seconds to prevent Free Tier RAM spikes
    max_samples = 10 * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        
    noisy_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced_tensor = model(noisy_tensor)
        
    enhanced = enhanced_tensor.squeeze().numpy()
    
    # Save output
    sf.write("temp_out.wav", enhanced, 16000)
    return "temp_out.wav"


# 5. UI Layout
st.title("AERO-MUTE: Real-Time Tactical Speech Enhancement")
st.markdown("Upload noisy communication audio (drone rotors, gunfire, wind). The FullSubNet+ architecture will decouple the interference in real-time.")

st.markdown("---")

# Drag & Drop Zone
uploaded_file = st.file_uploader("Upload Noisy Audio (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Incoming Signal (Noisy)")
        st.audio(uploaded_file)
        
    with col2:
        st.subheader("🎧 Processed Signal (Clean)")
        if st.button("Engage AERO-MUTE"):
            with st.spinner("Processing through Neural Subnets..."):
                out_file = process_audio(uploaded_file.getvalue())
                st.audio(out_file)
                st.success("Enhancement Complete.")

st.markdown("---")

# Pre-Generated Sample Results
st.header("Sample Operations Gallery")
st.markdown("Click play on any of the pre-processed mission clips below to instantly hear the 10+ dB noise reduction.")

# Hardcode the 5 examples we saved earlier
for i in range(1, 6):
    st.markdown(f"#### Mission Clip {i}")
    colA, colB, colC = st.columns(3)
    
    with colA:
        st.caption("Noisy Environment")
        st.audio(f"examples/clip_{i}_noisy.wav")
    with colB:
        st.caption("AERO-MUTE Enhanced")
        st.audio(f"examples/clip_{i}_enhanced.wav")
    with colC:
        st.caption("Ground Truth")
        st.audio(f"examples/clip_{i}_clean.wav")
    
    st.write("") # Spacing
