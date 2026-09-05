import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
import gradio as gr

# 1. Architecture Definition
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

# 2. Load Model
device = torch.device('cpu') # Hugging Face Free Tier uses CPU
model = FullSubNetPlus().to(device)
print("Loading model...")
checkpoint = torch.load("aeromute_fullsubnet_epoch_25.pt", map_location=device, weights_only=True)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()

# 3. Inference Function
def enhance_audio(audio_path):
    if audio_path is None:
        return None
    
    # Load and resample to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Cap at 10 seconds to prevent Free Tier timeouts
    max_samples = 10 * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]
        
    noisy_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced_tensor = model(noisy_tensor)
        
    enhanced = enhanced_tensor.squeeze().numpy()
    
    # Save output
    out_path = "enhanced_output.wav"
    sf.write(out_path, enhanced, 16000)
    return out_path

# 4. Custom CSS for Military/Tactical Theme
custom_css = """
body {
    background-color: #0a0a0a !important;
    background-image: radial-gradient(circle at 50% 0%, #1a2a1a 0%, #0a0a0a 70%);
}
.gradio-container {
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
}
.gr-button-primary {
    background: linear-gradient(135deg, #2e5c3a 0%, #1a3a22 100%) !important;
    border: 1px solid #4ade80 !important;
    box-shadow: 0 0 10px rgba(74, 222, 128, 0.2);
    transition: all 0.3s ease;
}
.gr-button-primary:hover {
    box-shadow: 0 0 20px rgba(74, 222, 128, 0.4);
    transform: translateY(-2px);
}
h1 {
    text-align: center;
    color: #4ade80;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(74,222,128,0.3);
}
.gr-panel {
    background: rgba(20, 25, 20, 0.7) !important;
    border: 1px solid #2e4a35 !important;
    backdrop-filter: blur(10px);
}
"""

# 5. Build Gradio UI
with gr.Blocks(css=custom_css, theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# AERO-MUTE: Real-Time Tactical Speech Enhancement")
    gr.Markdown("Upload noisy communication audio (drone rotors, gunfire, wind). The FullSubNet+ architecture will decouple the interference in real-time.")
    
    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(type="filepath", label="Input Signal (Microphone)")
            btn = gr.Button("Engage AERO-MUTE", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Cleaned Signal (Headset output)")
            
    btn.click(fn=enhance_audio, inputs=audio_in, outputs=audio_out)
    
    gr.Markdown("### Sample Operations")
    gr.Markdown("Click on any of the pre-processed mission clips below to instantly hear the Before and After results.")
    
    # Preload the 5 clips we generated
    examples = [
        ["examples/clip_1_noisy.wav"],
        ["examples/clip_2_noisy.wav"],
        ["examples/clip_3_noisy.wav"],
        ["examples/clip_4_noisy.wav"],
        ["examples/clip_5_noisy.wav"]
    ]
    
    gr.Examples(examples=examples, inputs=audio_in)

demo.launch()
