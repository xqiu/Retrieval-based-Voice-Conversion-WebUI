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
