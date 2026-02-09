"""
Simple configuration settings for AviaNZ model training.
Centralizes commonly used parameters to avoid magic numbers.
"""

DEFAULT_SAMPLE_RATE = 32000

# AST paper: 25ms window, 10ms hop -> 100 frames/second
DEFAULT_WINDOW_SECONDS = 0.025      # 25ms window
DEFAULT_HOP_SECONDS = 0.010         # 10ms hop
DEFAULT_FREQ_BINS = 224           # Final number of frequency bins (height)
DEFAULT_TIME_BINS = 1024            # Number of time bins (width) - 1024 bins = 10.24 seconds at 10ms hop (matches AudioSet pretraining)

SPECTROGRAM_PARAMS = {
    'windowType': 'Hann',
    'sgType': 'Standard', 
    'sgScale': 'Mel Frequency',
    'mean_normalise': False,
    'equal_loudness': False,
    'nfilters': DEFAULT_FREQ_BINS  # Number of frequency bins/filters for output
}

# Spectrogram transformation for model input
# We perform log compression once in the training pipeline if set to "Log".
# If you have externally pre-log-compressed spectrograms, change this to "None"
DEFAULT_SPEC_TRANSFORM = "Log"

# AST normalization statistics (from AudioSet training in the paper)
# Mean and std computed on log-mel spectrograms
AST_MEAN = -4.2677393
AST_STD = 4.5689974

# Training defaults
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_LEARNING_RATE = 3.073e-5  # Optuna-tuned optimal value (trial #18)
DEFAULT_CNN_LEARNING_RATE = 1e-3  # CNN models need higher learning rates
DEFAULT_CHANNELS = 1
DEFAULT_DROPOUT = 0.30  # Increased for better regularization with longer clips
DEFAULT_MIXUP_ALPHA = 0.25  # Reduced to match best practices (Kaytoo uses 0.25, spectrogram-level mixing)
DEFAULT_WEIGHT_DECAY = 1e-5  # Reduced slightly for AdamW
DEFAULT_BCE_SMOOTHING = 0.001  # Minimal smoothing to prevent overconfidence

# Data loading defaults
DEFAULT_MAX_SPECIES = 50
DEFAULT_MIN_EXAMPLES = 1000
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_VALIDATION_SHARE = 0.2
DEFAULT_NOISE_RATIO = 0.0  # Disabled by default to avoid accidental label corruption; enable explicitly when using a true noise-only folder
DEFAULT_NOISE_SAMPLES = 1000

# Spectrogram augmentation (applied during training)
DEFAULT_TIME_STRETCH_RANGE = (0.9, 1.1)  # ±10% time stretching
DEFAULT_PITCH_SHIFT_RANGE = (-2, 2)  # ±2 semitones pitch shifting
DEFAULT_FREQ_SHIFT_RANGE = (-10, 10)  # ±10 bins frequency shifting
DEFAULT_TEMPORAL_ROLL = True  # Enable temporal rolling augmentation (randomizes start position in tiled/repeated signals)

# Confusion sampling defaults (deprecated - confusion_sampling=False is optimal)
DEFAULT_CONFUSION_EVAL_FREQUENCY = 5
DEFAULT_CONFUSION_BOOST_FACTOR = 1.5
DEFAULT_CONFUSION_TOP_K = 10

# File processing limits (in seconds)
MAX_FILE_DURATION_SECONDS = 10000
WARNING_FILE_DURATION_SECONDS = 60  # 1 minute - warn if exceeded


def get_model_config():
    """
    Export configuration parameters needed for model deployment.
    Returns a dictionary containing all spectrogram and audio processing
    parameters that must match when loading the model for inference.
    
    This ensures that spectrograms are generated consistently during
    both training and inference.
    """
    return {
        # Audio processing
        'sample_rate': DEFAULT_SAMPLE_RATE,
        
        # Spectrogram time-based settings
        'window_seconds': DEFAULT_WINDOW_SECONDS,
        'hop_seconds': DEFAULT_HOP_SECONDS,
        'freq_bins': DEFAULT_FREQ_BINS,
        'time_bins': DEFAULT_TIME_BINS,
        
        # Spectrogram parameters
        'spectrogram_params': SPECTROGRAM_PARAMS.copy(),
        
        # Spectrogram transformation
        'spec_transform': DEFAULT_SPEC_TRANSFORM,
    }
