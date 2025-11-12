import argparse
import os
import sys
import traceback
import pathlib
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config
from infer.modules.vc.modules import VC

# 1. parser = argparse.ArgumentParser(description="Batch inference CLI using vc_multi for a directory of audio segments")

# Purpose: Initializes the argument parser with a description of what the script does.
# Explanation: The description provides a brief summary of the script’s functionality, which appears when the user runs the script with -h or --help. Here, it indicates the script processes a directory of audio segments using vc_multi, likely a voice conversion or synthesis model.

# 2. --sid

# Definition: parser.add_argument("--sid", type=int, default=0, help="speaker ID to use (default: 0)")
# Purpose: Specifies the speaker ID for voice conversion or synthesis.
# Explanation: In voice conversion or text-to-speech systems, models often support multiple speaker profiles. The --sid parameter (short for "speaker ID") selects which speaker’s voice characteristics to use. The default value is 0, meaning the first speaker in the model’s configuration is used if no other ID is specified. For example, if the model was trained on multiple voices, setting --sid 1 would select the second speaker.
# Example: --sid 2 selects speaker ID 2.

# 3. --dir_path

# Definition: parser.add_argument("--dir_path", type=str, required=True, help="input directory containing audio segments to process")
# Purpose: Specifies the input directory containing audio files to be processed.
# Explanation: This is a required argument (required=True) that points to a folder with audio segments (e.g., WAV, MP3 files) to be processed by the model. The script will likely iterate over all audio files in this directory for batch processing. For example, this could be a folder of voice recordings to be converted to another speaker’s voice.
# Example: --dir_path /path/to/audio_segments points to a directory containing audio files.

# 4. --opt_path

# Definition: parser.add_argument("--opt_path", type=str, required=True, help="output directory for converted audio files")
# Purpose: Specifies the output directory where processed audio files will be saved.
# Explanation: This required argument (required=True) defines where the script will save the converted or synthesized audio files after processing. The directory must exist or be created by the script. For example, if the script performs voice conversion, the converted audio files will be written to this folder.
# Example: --opt_path /path/to/output saves processed files to the specified directory.

# 5. --model_name

# Definition: parser.add_argument("--model_name", type=str, required=True, help="model name under assets/weight_root to load")
# Purpose: Specifies the name of the model to load for inference.
# Explanation: This required argument identifies the pre-trained model to use, located under a directory like assets/weight_root. For example, in a voice conversion system, this could be the name of a model checkpoint (e.g., model_v1.pth) trained for specific voice conversion tasks. The script will load this model for processing the audio files.
# Example: --model_name model_v1.pth loads the model file model_v1.pth from assets/weight_root.

# 6. --f0up_key

# Definition: parser.add_argument("--f0up_key", type=int, default=0, help="pitch shift in semitones (default: 0)")
# Purpose: Controls pitch shifting of the audio output.
# Explanation: The f0up_key parameter adjusts the fundamental frequency (F0, or pitch) of the output audio in semitones. A value of 0 (default) means no pitch shift. Positive values (e.g., 2) raise the pitch by that many semitones, while negative values (e.g., -2) lower it. This is useful in voice conversion to make the output voice sound higher or lower.
# Example: --f0up_key 4 raises the pitch by 4 semitones.

# 7. --f0method

# Definition: parser.add_argument("--f0method", type=str, default="harvest", help="pitch extraction algorithm: harvest or pm or others")
# Purpose: Specifies the algorithm used to extract pitch (F0) from the input audio.
# Explanation: Pitch extraction is critical in voice conversion to preserve or modify the pitch contour of the input audio. The default method is "harvest", a robust pitch estimation algorithm. Another option, "pm", might refer to a faster but less accurate method (e.g., Praat’s pitch estimation). Other methods may be supported depending on the vc_multi implementation. The choice affects the quality of pitch tracking in the processed audio.
# Example: --f0method pm uses the PM (Pitch Marking) algorithm for pitch extraction.

# 8. --index_path

# Definition: parser.add_argument("--index_path", type=str, default="", help="path to index file")
# Purpose: Specifies the path to an index file, likely used for speaker or feature lookup.
# Explanation: In voice conversion systems, an index file (e.g., a Faiss index) is often used to store precomputed speaker embeddings or features for efficient retrieval during inference. If left as an empty string (default), the script might not use an index or rely on a default one. This is optional, as not all models require an index file.
# Example: --index_path /path/to/index.faiss loads a specific index file.

# 9. --index2_path

# Definition: parser.add_argument("--index2_path", type=str, default="", help="alternative index path (optional)")
# Purpose: Specifies an alternative or secondary index file path.
# Explanation: Similar to --index_path, this allows specifying a second index file, possibly for a different set of speaker embeddings or features. It’s optional (default is empty), and its use depends on the model’s implementation. For example, it might be used for a fallback index or a different feature set.
# Example: --index2_path /path/to/index2.faiss loads an alternative index file.

# 10. --index_rate

# Definition: parser.add_argument("--index_rate", type=float, default=0.66, help="index retrieval ratio (default: 0.66)")
# Purpose: Controls the influence of the index file in the voice conversion process.
# Explanation: The index_rate determines how much the model relies on the index file (e.g., speaker embeddings) versus other features (like the input audio’s characteristics). A value of 0.66 (default) suggests a balanced mix, where 66% of the output is influenced by the index. A value of 0 would ignore the index, while 1 would fully rely on it. This is useful for fine-tuning the output voice’s similarity to the target speaker.
# Example: --index_rate 0.8 increases the index’s influence to 80%.

# 11. --filter_radius

# Definition: parser.add_argument("--filter_radius", type=int, default=3, help="filter radius for median filtering (default: 3)")
# Purpose: Specifies the radius for median filtering applied to pitch or other features.
# Explanation: Median filtering smooths out noise or outliers in extracted features like pitch (F0). The filter_radius defines the size of the window used for filtering (e.g., a radius of 3 considers 7 samples: the current sample and 3 samples on either side). A larger radius increases smoothing but may lose detail. The default (3) is a moderate setting.
# Example: --filter_radius 5 applies stronger smoothing to the pitch contour.

# 12. --resample_sr

# Definition: parser.add_argument("--resample_sr", type=int, default=0, help="resample sample rate after conversion (0=no resample)")
# Purpose: Specifies the sample rate for resampling the output audio.
# Explanation: After processing, the output audio can be resampled to a specific sample rate (in Hz). The default (0) means no resampling, so the output retains the input’s sample rate. For example, setting --resample_sr 16000 would resample the output to 16 kHz, which might be useful for compatibility or reducing file size.
# Example: --resample_sr 44100 resamples the output to 44.1 kHz.

# 13. --rms_mix_rate

# Definition: parser.add_argument("--rms_mix_rate", type=float, default=1.0, help="RMS envelope mix ratio (default: 1.0)")
# Purpose: Controls the blending of the RMS (Root Mean Square) envelope between input and target audio.
# Explanation: The RMS envelope represents the loudness or amplitude contour of the audio. The rms_mix_rate determines how much of the input audio’s RMS envelope is retained versus the target’s. A value of 1.0 (default) likely uses the target’s RMS envelope fully, while 0.0 would use the input’s. Values in between (e.g., 0.5) blend the two, affecting the output’s loudness dynamics.
# Example: --rms_mix_rate 0.5 blends the input and target RMS envelopes equally.

# 14. --protect

# Definition: parser.add_argument("--protect", type=float, default=0.33, help="protection for consonants and breaths (default: 0.33)")
# Purpose: Controls how much consonants and breaths are preserved in the output.
# Explanation: In voice conversion, consonants and breaths (non-voiced sounds) can be challenging to convert naturally. The protect parameter adjusts how much these elements are preserved from the input audio versus being altered by the model. A value of 0.33 (default) suggests moderate protection, balancing naturalness and conversion quality. Higher values (e.g., 0.5) preserve more of the input’s consonants and breaths.
# Example: --protect 0.5 increases protection for consonants and breaths.

# 15. --format

# Definition: parser.add_argument("--format", type=str, default="wav", choices=["wav","flac","mp3","m4a"], help="output file format (default: wav)")
# Purpose: Specifies the file format for the output audio files.
# Explanation: The processed audio files can be saved in one of the supported formats: WAV (lossless, default), FLAC (lossless, compressed), MP3 (lossy), or M4A (lossy). The choice affects file size and compatibility. WAV is the default for high-quality, uncompressed audio.
# Example: --format mp3 saves output files as MP3s.

# 16. --device

# Definition: parser.add_argument("--device", type=str, help="CUDA device specification, e.g. '0' or '0,1'")
# Purpose: Specifies the CUDA device(s) for GPU-accelerated inference.
# Explanation: This optional argument selects which GPU(s) to use for computation, identified by their CUDA device IDs (e.g., 0 for the first GPU, 0,1 for multiple GPUs). If not provided, the script might default to CPU or a single GPU. This is relevant for users with NVIDIA GPUs and CUDA installed.
# Example: --device 0 uses the first GPU.

# 17. --is_half

# Definition: parser.add_argument("--is_half", type=bool, help="use half precision: True or False")
# Purpose: Enables or disables half-precision (e.g., FP16) computation.
# Explanation: Half-precision reduces memory usage and speeds up inference on compatible hardware (e.g., modern NVIDIA GPUs). If set to True, the model uses FP16; if False, it uses full precision (FP32). The default isn’t specified, so it depends on the script’s implementation. This is useful for optimizing performance on resource-constrained systems.
# Example: --is_half True enables half-precision computation.

# 18. args = parser.parse_args()

# Purpose: Parses the command-line arguments and stores them in the args object.
# Explanation: This line processes the user’s command-line input, validates it against the defined arguments, and creates an args object with attributes corresponding to each parameter (e.g., args.sid, args.dir_path). The script uses these values to configure the inference process.

def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference CLI using vc_multi for a directory of audio segments")
    parser.add_argument("--sid", type=int, default=0, help="speaker ID to use (default: 0)")
    parser.add_argument("--dir_path", type=str, required=True, help="input directory containing audio segments to process")
    parser.add_argument("--opt_path", type=str, required=True, help="output directory for converted audio files")
    parser.add_argument("--model_name", type=str, required=True, help="model name under assets/weight_root to load")
    parser.add_argument("--f0up_key", type=int, default=0, help="pitch shift in semitones (default: 0)")
    parser.add_argument("--f0method", type=str, default="harvest", help="pitch extraction algorithm: harvest or pm or others")
    parser.add_argument("--index_path", type=str, default="", help="path to index file")
    parser.add_argument("--index2_path", type=str, default="", help="alternative index path (optional)")
    parser.add_argument("--index_rate", type=float, default=0.66, help="index retrieval ratio (default: 0.66)")
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius for median filtering (default: 3)")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sample rate after conversion (0=no resample)")
    parser.add_argument("--rms_mix_rate", type=float, default=1.0, help="RMS envelope mix ratio (default: 1.0)")
    parser.add_argument("--protect", type=float, default=0.33, help="protection for consonants and breaths (default: 0.33)")
    parser.add_argument("--format", type=str, default="wav", choices=["wav","flac","mp3","m4a"], help="output file format (default: wav)")
    parser.add_argument("--device", type=str, help="CUDA device specification, e.g. '0' or '0,1'")
    parser.add_argument("--is_half", type=bool, help="use half precision: True or False")
    args = parser.parse_args()
    # reset argv so downstream libs don't get confused
    sys.argv = sys.argv[:1]
    return args


def main():
    load_dotenv()
    args = arg_parse()
    config = Config()
    # apply optional device and precision settings
    if args.device:
        config.device = args.device
    if args.is_half is not None:
        config.is_half = args.is_half
    vc = VC(config)
    # load the specified model
    vc.get_vc(args.model_name)
    # batch infer on the directory, with error tracing
    try:
        for info in vc.vc_multi(
            args.sid,
            args.dir_path,
            args.opt_path,
            None,
            args.f0up_key,
            args.f0method,
            args.index_path,
            args.index2_path,
            args.index_rate,
            args.filter_radius,
            args.resample_sr,
            args.rms_mix_rate,
            args.protect,
            args.format,
        ):
            print(info)
    except Exception as e:
        # Provide detailed context and traceback
        print(f"Error during batch inference in directory: {args.dir_path}", file=sys.stderr)
        print(f"Model: {args.model_name}, Index: {args.index_path}, Params: sid={args.sid}, f0method={args.f0method}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
