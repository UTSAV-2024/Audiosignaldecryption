import streamlit as st
import numpy as np
import wave
from scipy.io import wavfile
import plotly.graph_objects as go
from scipy.signal import spectrogram
import tempfile
import os
import math

# -----------------------------
# Helper functions
# -----------------------------

def load_audio_from_upload(uploaded_file):
    """Load audio from uploaded file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        sample_rate, audio = wavfile.read(tmp_file_path)
        os.unlink(tmp_file_path)
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]
            
        return audio, sample_rate
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def plot_interactive_spectrogram(audio, sample_rate, freq_min=0, freq_max=200000):
    try:
        f, t, Sxx = spectrogram(audio, fs=sample_rate, nperseg=4096, noverlap=3072)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        target_min = 2500
        target_max = 5500
        is_target_range = (freq_min <= target_min + 500 and freq_max >= target_max - 500)

        if not is_target_range:
            hidden_freq_start = 2700
            hidden_freq_end = 5300
            hidden_freq_mask = (f >= hidden_freq_start) & (f <= hidden_freq_end)

            if freq_max < target_min or freq_min > target_max:
                Sxx_db[hidden_freq_mask, :] = Sxx_db[hidden_freq_mask, :] * 0.05 + np.random.normal(-25, 4, Sxx_db[hidden_freq_mask, :].shape)
            else:
                Sxx_db[hidden_freq_mask, :] = Sxx_db[hidden_freq_mask, :] * 0.4 + np.random.normal(-8, 3, Sxx_db[hidden_freq_mask, :].shape)

        freq_mask = (f >= freq_min) & (f <= freq_max)
        f_filtered = f[freq_mask]
        Sxx_filtered = Sxx_db[freq_mask, :]

        if is_target_range:
            Sxx_filtered = Sxx_filtered * 1.3

        fig = go.Figure(data=go.Heatmap(
            z=Sxx_filtered,
            x=t,
            y=f_filtered,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)"),
            zmin=np.percentile(Sxx_filtered, 5) if len(Sxx_filtered) > 0 else -50,
            zmax=np.percentile(Sxx_filtered, 95) if len(Sxx_filtered) > 0 else 0
        ))

        fig.update_layout(
            title=f"Audio Spectrogram - Range: {freq_min:,} - {freq_max:,} Hz",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            height=600,
            yaxis=dict(range=[freq_min, freq_max])
        )

        return fig, is_target_range
    except Exception as e:
        st.error(f"Error creating spectrogram: {str(e)}")
        return None, False

def analyze_frequency_content(audio, sample_rate, freq_min, freq_max):
    try:
        f, t, Sxx = spectrogram(audio, fs=sample_rate, nperseg=4096, noverlap=3072)
        freq_mask = (f >= freq_min) & (f <= freq_max)
        target_freqs = f[freq_mask]
        target_power = Sxx[freq_mask, :]

        if len(target_freqs) == 0 or len(target_power) == 0:
            return 0, [], [], []

        avg_power = np.mean(target_power, axis=1)
        background_threshold = np.percentile(avg_power, 55) if len(avg_power) > 0 else 0
        peak_indices = np.where(avg_power > background_threshold)[0] if len(avg_power) > 0 else []
        peak_freqs = target_freqs[peak_indices] if len(peak_indices) > 0 else []

        estimated_characters = 0
        if len(peak_freqs) > 0:
            freq_groups = []
            current_group = [peak_freqs[0]]

            for freq in peak_freqs[1:]:
                if freq - current_group[-1] < 200:
                    current_group.append(freq)
                else:
                    freq_groups.append(current_group)
                    current_group = [freq]
            freq_groups.append(current_group)

            estimated_characters = len(freq_groups)

        return estimated_characters, peak_freqs, target_freqs, avg_power
    except Exception as e:
        st.error(f"Error analyzing frequency content: {str(e)}")
        return 0, [], [], []

def detect_hidden_message(audio, sample_rate):
    target_min = 2700
    target_max = 5300
    try:
        char_count, peak_freqs, _, _ = analyze_frequency_content(audio, sample_rate, target_min, target_max)
        if char_count >= 2:
            return True, char_count, peak_freqs
        return False, char_count, peak_freqs
    except Exception:
        return False, 0, []

# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(page_title="🔍 Audio Signal Decrypter", layout="wide")

st.title("🔍 Audio Signal Decrypter - 200kHz Challenge")
st.markdown("**Upload a WAV file and hunt through the 200,000 Hz spectrum to discover hidden messages!**")

if 'puzzle_unlocked' not in st.session_state:
    st.session_state.puzzle_unlocked = False

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

# -----------------
# PHASE 1: TOUGH PUZZLE
# -----------------
if not st.session_state.puzzle_unlocked:
    st.subheader("🌱 Unlock the Sustainability Puzzle")

st.markdown("""
Imagine you are a **city planner** working to combat climate change!  
Two major green initiatives are underway:

1. **Eco-Park Development** 🌳 – reduces carbon footprint by $c_1$ metric tons/year.  
2. **Green Transit Expansion** 🚊 – reduces carbon footprint by $c_2$ metric tons/year.

The city has collected the following data:  

- The **combined carbon reduction** is **8,000 metric tons/year**.  
- The **greatest common divisor** (GCD) of $c_1$ and $c_2$ is **1,000 metric tons/year**.  
- The **sustainability synergy index** (product) is product of carbon footprint indexes which is 15,000,000 in this case.  
- The Eco-Park project must contribute **less than 60%** of the total reduction.  
- Both reductions are **multiples of 1,000 metric tons**.  

**Question:**  
Determine the **carbon reduction of each project individually** to unlock the hidden message! 🔑
""")

f1 = st.number_input("Enter Eco-Park Reduction $c_1$ (metric tons/year):", step=1.0, format="%.0f", key="f1")
f2 = st.number_input("Enter Green Transit Reduction $c_2$ (metric tons/year):", step=1.0, format="%.0f", key="f2")

if st.button("Submit Answer"):
    correct_sorted = sorted([3000, 5000])
    user_sorted = sorted([int(round(f1)), int(round(f2))])

    if user_sorted == correct_sorted:
        st.session_state.puzzle_unlocked = True
        st.success("✅ Correct! Sliders unlocked and the sustainability message is revealed!")
        st.balloons()
        st.rerun()
    else:
        st.error("❌ Incorrect. Try again!")

# -----------------
# PHASE 2: SPECTROGRAM
# -----------------
if uploaded_file is not None and st.session_state.puzzle_unlocked:
    audio, sample_rate = load_audio_from_upload(uploaded_file)

    if audio is not None and sample_rate is not None:
        st.success(f"✅ Audio loaded! Duration: {len(audio)/sample_rate:.2f}s | Sample Rate: {sample_rate:,} Hz")
        st.audio(uploaded_file, format="audio/wav")

        freq_min = st.slider("Minimum Frequency (Hz)", 0, 199000, 0, 1000)
        freq_max = st.slider("Maximum Frequency (Hz)", 1000, 200000, 200000, 1000)

        if freq_min < freq_max:
            fig, is_in_target_range = plot_interactive_spectrogram(audio, sample_rate, freq_min, freq_max)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

                is_hidden, hidden_chars, hidden_peak_freqs = detect_hidden_message(audio, sample_rate)
                perfect_range = (freq_min <= 2700 and freq_max >= 5300)

                if perfect_range and is_in_target_range:
                    if is_hidden and hidden_chars > 0:
                        st.success("🎉 JACKPOT! COMPLETE MESSAGE DECODED!")
                        st.balloons()
