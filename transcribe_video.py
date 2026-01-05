import sys
import os
from pathlib import Path
from faster_whisper import WhisperModel


# ============================================================================
# CONFIGURATION PARAMETERS - Adjust these as needed
# ============================================================================

# Model settings
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large-v2, large-v3
DEVICE = "cpu"  # Options: "cpu" or "cuda" (for GPU)
COMPUTE_TYPE = "int8"  # For CPU: int8 | For GPU: float16, int8_float16

# Transcription settings
LANGUAGE = "ar"  # Language code (ar=Arabic, en=English, auto for auto-detect)
BEAM_SIZE = 5  # Higher = more accurate but slower (1-10)
VAD_FILTER = True  # Voice Activity Detection to filter silence
VAD_MIN_SILENCE_MS = 500  # Minimum silence duration in milliseconds

# ============================================================================


def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python transcribe_video.py <video_path>")
        print("Example: python transcribe_video.py '/path/to/video.mp4'")
        sys.exit(1)
    
    # Get video path from command line
    video_path = sys.argv[1]
    
    # Validate video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Generate output file path (same location, same name with _transcript.txt)
    video_file = Path(video_path)
    output_file = video_file.parent / f"{video_file.stem}_transcript.txt"
    
    print("="*80)
    print("FASTER-WHISPER TRANSCRIPTION")
    print("="*80)
    print(f"Video file:    {video_path}")
    print(f"Output file:   {output_file}")
    print(f"Model:         {MODEL_SIZE}")
    print(f"Device:        {DEVICE}")
    print(f"Language:      {LANGUAGE}")
    print(f"Beam size:     {BEAM_SIZE}")
    print("="*80)
    print("\nLoading model...")
    
    # Initialize the model
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    print("Model loaded successfully!")
    print("Starting transcription (this may take a few minutes)...\n")
    
    # Transcribe the video
    segments, info = model.transcribe(
        video_path,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
    )
    
    print(f"Detected language: '{info.language}' (probability: {info.language_probability:.2f})")
    print(f"\nTranscription:\n{'='*80}")
    
    # Save transcription to file and display it
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            timestamp = f"[{segment.start:.2f}s -> {segment.end:.2f}s]"
            text = segment.text.strip()
            line = f"{timestamp} {text}"
            print(line)
            f.write(line + "\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Transcription completed!")
    print(f"✓ Saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
