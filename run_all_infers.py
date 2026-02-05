import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def sanitize_name(value: str) -> str:
	return value.replace(os.sep, "_").replace(":", "_").replace(" ", "_")


def format_index_rate(index_rate: float) -> str:
	text = f"{index_rate:.3f}".rstrip("0").rstrip(".")
	return text.replace(".", "p")


def build_output_path(
	output_dir: Path,
	input_path: Path,
	output_ext: str | None,
	model_name: str | None = None,
	index_rate: float | None = None,
) -> Path:
	if output_ext:
		output_ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
	else:
		output_ext = input_path.suffix

	name_parts = [input_path.stem]
	if model_name:
		name_parts.append(sanitize_name(model_name))
	if index_rate is not None:
		name_parts.append(f"ir{format_index_rate(index_rate)}")

	return output_dir / f"{'__'.join(name_parts)}{output_ext}"


def iter_input_files(input_dir: Path, recursive: bool, extensions: set[str] | None):
	pattern = "**/*" if recursive else "*"
	for path in input_dir.glob(pattern):
		if not path.is_file():
			continue
		if extensions and path.suffix.lower() not in extensions:
			continue
		yield path


def run_infer(
	python_exe: str,
	input_path: Path,
	output_path: Path,
	index_path: str,
	model: str,
	f0method: str,
	f0up_key: int,
	index_rate: float,
	device: str,
	is_half: bool,
	extra_args: list[str],
):
	cmd = [
		python_exe,
		"tools/infer_cli.py",
		"--f0up_key",
		str(f0up_key),
		"--input_path",
		str(input_path),
		"--index_path",
		index_path,
		"--model",
		model,
		"--f0method",
		f0method,
		"--opt_path",
		str(output_path),
		"--index_rate",
		str(index_rate),
		"--device",
		device,
		"--is_half",
		"True" if is_half else "False",
	]
	cmd.extend(extra_args)
	subprocess.run(cmd, check=True)


def load_settings(settings_path: Path) -> list[dict]:
	with settings_path.open("r", encoding="utf-8") as file:
		data = json.load(file)

	if isinstance(data, list):
		settings = data
	elif isinstance(data, dict) and isinstance(data.get("models"), list):
		settings = data["models"]
	else:
		raise SystemExit("settings_json must be a list or contain a 'models' list.")

	for idx, item in enumerate(settings, start=1):
		if not isinstance(item, dict):
			raise SystemExit(f"settings_json item {idx} must be an object.")
		if "model" not in item or "index_path" not in item:
			raise SystemExit(f"settings_json item {idx} must include 'model' and 'index_path'.")
	return settings


def main() -> int:
	parser = argparse.ArgumentParser(description="Batch run infer_cli.py for all files in a directory.")
	parser.add_argument("--input_dir", required=True, help="Directory containing input audio files.")
	parser.add_argument("--output_dir", required=True, help="Directory to write outputs.")
	parser.add_argument("--index_path", help="Path to index file.")
	parser.add_argument("--model", help="Model filename or path.")
	parser.add_argument("--f0method", default="rmvpe", help="F0 method, e.g. rmvpe.")
	parser.add_argument("--f0up_key", type=int, default=0, help="F0 up key.")
	parser.add_argument("--index_rate", type=float, default=0.6, help="Index rate.")
	parser.add_argument("--device", default="cuda:0", help="Device string, e.g. cuda:0.")
	parser.add_argument("--is_half", action="store_true", help="Use half precision.")
	parser.add_argument("--python_exe", default=sys.executable, help="Python executable to run.")
	parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
	parser.add_argument(
		"--extensions",
		default=".wav,.flac,.mp3,.m4a,.ogg,.aac",
		help="Comma-separated list of input extensions to include.",
	)
	parser.add_argument("--output_ext", default="", help="Force output extension, e.g. wav.")
	parser.add_argument(
		"--settings_json",
		default="",
		help="Path to JSON settings file defining models and indexes.",
	)
	parser.add_argument(
		"--extra_args",
		nargs=argparse.REMAINDER,
		default=[],
		help="Extra args passed to infer_cli.py (put after --extra_args).",
	)

	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	if not input_dir.is_dir():
		raise SystemExit(f"input_dir not found: {input_dir}")
	output_dir.mkdir(parents=True, exist_ok=True)

	extensions = None
	if args.extensions:
		extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
		if "" in extensions:
			extensions.remove("")

	output_ext = args.output_ext.strip() or None
	settings = None
	if args.settings_json:
		settings = load_settings(Path(args.settings_json))
	else:
		if not args.model or not args.index_path:
			raise SystemExit("--model and --index_path are required when --settings_json is not provided.")
		settings = [
			{
				"model": args.model,
				"index_path": args.index_path,
				"index_rate": args.index_rate,
			}
		]

	files = list(iter_input_files(input_dir, args.recursive, extensions))
	if not files:
		print("No input files found.")
		return 0

	for input_path in files:
		for setting in settings:
			model = setting["model"]
			index_path = setting["index_path"]
			index_rate = float(setting.get("index_rate", args.index_rate))
			model_name = setting.get("name") or Path(model).stem
			output_path = build_output_path(
				output_dir,
				input_path,
				output_ext,
				model_name=model_name,
				index_rate=index_rate,
			)
			output_path.parent.mkdir(parents=True, exist_ok=True)
			run_infer(
				python_exe=args.python_exe,
				input_path=input_path,
				output_path=output_path,
				index_path=index_path,
				model=model,
				f0method=args.f0method,
				f0up_key=args.f0up_key,
				index_rate=index_rate,
				device=args.device,
				is_half=args.is_half,
				extra_args=args.extra_args,
			)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())