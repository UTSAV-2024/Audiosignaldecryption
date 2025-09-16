import numpy as np
import wave
from scipy.io import wavfile
import os

def create_letter_patterns():
    """
    Define simple letter patterns for spectrogram display.
    Each letter is a 7x5 grid (frequency x time)
    1 = tone present, 0 = no tone
    NOTE: Patterns are defined normally but will be flipped for spectrogram display
    """
    patterns = {
        'F': [
            [1,0,0,0,0],  # Bottom row (lowest frequency)
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,1,1,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,1,1,1,1]   # Top row (highest frequency)
        ],
        'L': [
            [1,1,1,1,1],  # Bottom row (lowest frequency)
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0]   # Top row (highest frequency)
        ],
        'A': [
            [1,0,0,0,1],  # Bottom row
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,1,1,1,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0]   # Top row
        ],
        'G': [
            [0,1,1,1,0],  # Bottom row
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,0,1,1,1],
            [1,0,0,0,0],
            [1,0,0,0,1],
            [0,1,1,1,0]   # Top row
        ],
        '4': [
            [0,0,0,1,0],  # Bottom row
            [0,0,0,1,0],
            [0,0,0,1,0],
            [1,1,1,1,1],
            [1,0,0,1,0],
            [1,0,0,1,0],
            [1,0,0,1,0]   # Top row
        ],
        '2': [
            [1,1,1,1,1],  # Bottom row
            [1,0,0,0,0],
            [1,0,0,0,0],
            [0,1,1,1,0],
            [0,0,0,0,1],
            [0,0,0,0,1],
            [1,1,1,1,0]   # Top row
        ]
    }
    return patterns

def generate_base_audio(duration=10, sample_rate=44100):
    """
    Generate a simple base audio (white noise or sine wave)
    In practice, you'd load your actual song here
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple background: low-frequency sine wave + gentle noise
    background = 0.1 * np.sin(2 * np.pi * 200 * t)  # 200Hz base tone
    background += 0.05 * np.random.normal(0, 1, len(t))  # Light noise
    
    return background, t, sample_rate

def embed_message_in_spectrogram(audio, t, message, sample_rate=44100):
    """
    Embed a text message in the audio spectrogram
    """
    patterns = create_letter_patterns()
    modified_audio = audio.copy()
    
    # Parameters for the hidden message
    base_freq = 4000  # Starting frequency (Hz)
    freq_step = 100   # Frequency step between rows
    time_step = 0.3   # Time step between columns (seconds)
    char_spacing = 0.8  # Space between characters (seconds)
    amplitude = 0.02  # Volume of hidden tones (quiet but visible)
    
    start_time = 2.0  # Start hiding message at 2 seconds
    
    for char_idx, char in enumerate(message.upper()):
        if char not in patterns:
            continue
            
        pattern = patterns[char]
        char_start_time = start_time + char_idx * (5 * time_step + char_spacing)
        
        # For each row in the pattern (frequency)
        for row_idx, row in enumerate(pattern):
            freq = base_freq + row_idx * freq_step
            
            # For each column in the pattern (time)
            for col_idx, pixel in enumerate(row):
                if pixel == 1:  # Draw this pixel
                    tone_start_time = char_start_time + col_idx * time_step
                    tone_duration = time_step * 0.8  # Slightly shorter for clarity
                    
                    # Find the time indices for this tone
                    start_idx = int(tone_start_time * sample_rate)
                    end_idx = int((tone_start_time + tone_duration) * sample_rate)
                    
                    if end_idx < len(modified_audio):
                        # Generate sine wave for this pixel
                        tone_t = t[start_idx:end_idx]
                        tone = amplitude * np.sin(2 * np.pi * freq * tone_t)
                        
                        # Add to the audio
                        modified_audio[start_idx:end_idx] += tone
    
    return modified_audio

def save_audio(audio, filename, sample_rate=44100):
    """
    Save audio to WAV file
    """
    # Normalize audio to prevent clipping
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

def analyze_audio_spectrogram(filename):
    """
    Load and display the spectrogram - this is what participants would do
    """
    sample_rate, audio = wavfile.read(filename)
    
    # Create spectrogram
    plt.figure(figsize=(12, 8))
    plt.specgram(audio, Fs=sample_rate, NFFT=1024, noverlap=512)
    plt.colorbar(label='Power (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Audio Spectrogram - Look for Hidden Message!')
    plt.ylim(3000, 5000)  # Focus on the frequency range where message is hidden
    plt.show()
    
    return "Check the spectrogram display - can you see any patterns?"

def main_example():
    """
    Complete example: Create audio with hidden message
    """
    print("🎵 Audio Steganography Puzzle Generator 🎵")
    print("=" * 50)
    
    # Step 1: Generate base audio (in practice, load your song)
    print("1. Generating base audio...")
    audio, t, sample_rate = generate_base_audio(duration=15)
    
    # Step 2: Choose your hidden message
    hidden_message = "FLAG42"  # Your puzzle answer
    print(f"2. Hiding message: '{hidden_message}'")
    
    # Step 3: Embed the message
    print("3. Embedding message in spectrogram...")
    modified_audio = embed_message_in_spectrogram(audio, t, hidden_message, sample_rate)
    
    # Step 4: Save the puzzle file
    output_file = "puzzle_audio.wav"
    save_audio(modified_audio, output_file, sample_rate)
    print(f"4. Saved puzzle audio as: {output_file}")
    
    # Step 5: Show what participants need to do
    print("\n🔍 How participants solve this:")
    print("1. Load 'puzzle_audio.wav' in Audacity")
    print("2. Select the audio track")
    print("3. Go to: Analyze → Plot Spectrum OR")
    print("   Change track view to 'Spectrogram'")
    print("4. Look for text patterns in 3000-5000 Hz range")
    print("5. Read the hidden message!")
    
    # Demonstrate the solution
    print(f"\n✅ Solution: The hidden message is '{hidden_message}'")
    
    return output_file

# Example of what participants would run to analyze
def participant_analysis_example():
    """
    This is what participants might do to analyze the audio
    """
    print("\n🕵 Participant Analysis Example:")
    print("=" * 40)
    
    # Load the audio file
    try:
        filename = "puzzle_audio.wav"
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            result = analyze_audio_spectrogram(filename)
            print(result)
        else:
            print("Run main_example() first to generate the puzzle audio!")
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        print("Make sure you have matplotlib and scipy installed!")

if __name__ == "__main__":
    # Generate the puzzle
    puzzle_file = main_example()
    
    print(f"\n🎯 Your puzzle is ready!")
    print(f"File created: {puzzle_file}")
    print(f"Location: {os.path.abspath(puzzle_file)}")
    print("\nGive this file to participants and tell them:")
    print("'This audio file contains a hidden message. Find it!'")
    print("\n💡 Hint for you: The message is hidden in the spectrogram")
    print("   Participants should analyze it with audio software like Audacity")
    
    # Participant analysis
    participant_analysis_example()