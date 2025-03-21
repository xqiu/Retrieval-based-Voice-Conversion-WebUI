import datetime
import os
import subprocess
import argparse
import sys
import shutil
import re

RVC_PATH = os.path.dirname(os.path.realpath(__file__))
VENV_PATH = os.path.join(RVC_PATH, ".venv", "python.exe")
LOG_FILE = os.path.join(RVC_PATH, "run_infer.log")
log_file = None

SPLIT_FAILED = "SPLIT_FAILED"
PROCESS_FAILED = "PROCESS_FAILED"
CONCAT_FAILED = "CONCAT_FAILED"

def log_message(message, level="INFO", log_file=LOG_FILE):
    if log_file is LOG_FILE:
        print(f"[WARNING] using default log file")
    with open(log_file, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{level}] {message}\n")

# 5 minutes in seconds
DEFAULT_MIN_SEGMENT_DURATION = 300
# 10 minutes in seconds
DEFAULT_MAX_SEGMENT_DURATION = 600
# Noise threshold in dB
DEFAULT_NOISE_THRESHOLD = -50
# Minimum silence duration in seconds
DEFAULT_SILENCE_DURATION = 0.5  

SUPPORTED_EXTENSION_LIST = ('.wav', '.mp3')

def get_silient_times(input_file, noise_threshold, silence_duration):
    global log_file

    silence_cmd = [
        'ffmpeg', '-i', input_file, '-af',
        f'silencedetect=noise={noise_threshold}dB:d={silence_duration}', '-f', 'null', '-'
    ]
    try:
        result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        log_message(f"{SPLIT_FAILED} detecting silence: {e}", level="ERROR", log_file=log_file)
        return False

    # Parse the output to find silence timestamps
    silence_times = []
    for line in result.stderr.split('\n'):
        if 'silence_end' in line:
            match = re.search(r'silence_end: (\d+\.?\d*)', line)
            if match:
                silence_times.append(float(match.group(1)))
    return silence_times

def split_audio_on_silence(input_file, output_dir, min_duration=DEFAULT_MIN_SEGMENT_DURATION, max_duration=DEFAULT_MAX_SEGMENT_DURATION, noise_threshold=DEFAULT_NOISE_THRESHOLD, silence_duration=DEFAULT_SILENCE_DURATION, enable = True):
    """
    Split an audio file into segments based on silence, with min and max duration constraints.

    Parameters:
    - input_file (str): Path to the input audio file (e.g., 'input.wav').
    - output_dir (str): Directory to save the split audio files.
    - min_duration (int): Minimum segment duration in seconds (default: 300 = 5 minutes).
    - max_duration (int): Maximum segment duration in seconds (default: 600 = 10 minutes).
    - noise_threshold (int): Noise threshold in dB for silence detection (default: -30).
    - silence_duration (float): Minimum silence duration in seconds (default: 0.5).
    - enable (bool): Enable or disable the splitting process (default: True).

    Returns:
    - bool: True if splitting was successful, False otherwise.
    """
    global log_file
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Detect silence timestamps
    silence_times = get_silient_times(input_file, noise_threshold, silence_duration) if enable else []
    #BEGIN NO SILENCE DETECTED
    if not silence_times:
        if enable:
            log_message("No silence detected. Adjust noise threshold or silence duration.", log_file=log_file)
        # Fallback: Copy the entire file as a single segment
        output_file = os.path.join(output_dir, '000.wav')
        ffmpeg_copy_cmd = [
            'ffmpeg', '-i', input_file, '-c:a', 'pcm_s16le', output_file
        ]
        try:
            subprocess.run(ffmpeg_copy_cmd, check=True, encoding='utf-8')
            log_message(f"No splits applied. Copied entire file to {output_file}", log_file=log_file)
            return True
        except subprocess.CalledProcessError as e:
            log_message(f"copying file: {e}", level="ERROR", log_file=log_file)
            return False
    #END NO SILENCE DETECTED
    # Step 2: Filter timestamps to enforce min/max duration
    filtered_split_times = []
    last_split_time = 0  # Start of the first segment

    for silence_time in silence_times:
        segment_duration = silence_time - last_split_time

        # If the segment would be shorter than the minimum duration, skip this split point
        if segment_duration < min_duration:
            continue

        # If the segment would be longer than the maximum duration, force a split at max_duration
        while segment_duration > max_duration:
            last_split_time += max_duration
            filtered_split_times.append(last_split_time)
            segment_duration = silence_time - last_split_time

        # Now the segment is within bounds (or close), so add the silence point as a split
        filtered_split_times.append(silence_time)
        last_split_time = silence_time

    # Step 3: Handle the last segment (if any audio remains)
    # Get the total duration of the audio
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        total_duration = float(subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True, check=True, encoding='utf-8').stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        log_message(f"{SPLIT_FAILED} getting audio duration: {e}", level="ERROR", log_file=log_file)
        return False

    # Check if there's remaining audio after the last split
    remaining_duration = total_duration - last_split_time
    while remaining_duration > 0:
        if remaining_duration > max_duration:
            # If the remaining audio is longer than the max duration, split at max_duration
            last_split_time += max_duration
            filtered_split_times.append(last_split_time)
            remaining_duration = total_duration - last_split_time
        elif remaining_duration < min_duration and len(filtered_split_times) > 0:
            # If the remaining audio is shorter than the min duration, merge it with the previous segment
            filtered_split_times.pop()  # Remove the last split point
            break
        else:
            # The remaining audio is within bounds, so no further splits are needed
            break

    # Step 4: Split the audio at the filtered timestamps
    if filtered_split_times:
        ffmpeg_split_cmd = [
            'ffmpeg', '-i', input_file, '-f', 'segment',
            '-segment_times', ','.join(map(str, filtered_split_times)),
            '-c:a', 'pcm_s16le', os.path.join(output_dir, '%03d.wav')
        ]
        try:
            subprocess.run(ffmpeg_split_cmd, check=True, encoding='utf-8')
            log_message(f"Split completed. Files saved in {output_dir}", log_file=log_file)
            return True
        except subprocess.CalledProcessError as e:
            log_message(f"{SPLIT_FAILED} splitting audio: {e}", level="ERROR", log_file=log_file)
            return False
    else:
        # If no split points were found, copy the entire file
        output_file = os.path.join(output_dir, '000.wav')
        ffmpeg_copy_cmd = [
            'ffmpeg', '-i', input_file, '-c:a', 'pcm_s16le', output_file
        ]
        try:
            subprocess.run(ffmpeg_copy_cmd, check=True, encoding='utf-8')
            log_message(f"No valid split points found. Copied entire file to {output_file}", log_file=log_file)
            return True
        except subprocess.CalledProcessError as e:
            log_message(f"copying file: {e}", level="ERROR", log_file=log_file)
            return False

def concat_audio_files(input_file_list, output_path):
    """
    Concatenate multiple audio files into a single file.

    Parameters:
    - input_file_list (list): List of input audio files to concatenate.
    - output_path (str): Path to save the concatenated audio file.

    Returns:
    - bool: True if concatenation was successful, False otherwise.
    """
    global log_file
    concat_list_path = os.path.join(os.path.dirname(output_path), f'concat_{os.path.basename(output_path)}.txt')
    with open(concat_list_path, 'w') as f:
        for input_file in input_file_list:
            f.write(f"file '{input_file}'\n")

    ffmpeg_concat_cmd = [
        'ffmpeg', 
        #auto confirm overwrite
        '-y',
        '-f', 'concat', 
        #since we use full path, we can use unsafe 0
        '-safe', '0', 
        '-i', concat_list_path, 
        '-c', 'copy', 
        output_path
    ]
    log_message(f'Concatenating audio files: {ffmpeg_concat_cmd}', log_file=log_file)
    try:
        subprocess.run(ffmpeg_concat_cmd, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        log_message(f"{CONCAT_FAILED}: {e}", level="ERROR", log_file=log_file)
        return False
    log_message(f"Concatenation completed. File saved as {output_path}", log_file=log_file)
    return True
    
def process_audio(source_path: str, output_path: str, model: str, index_path: str, index_rate: float, f0up_key: int, f0_method: str, gpu: str, support_f16: bool):
    global log_file
    cmd = [
        sys.executable, 'tools/infer_cli.py',
        '--f0up_key', f'{f0up_key}',
        '--input_path', source_path,
        '--model_name', model,
        '--device', gpu,
        '--is_half', f'{support_f16}',
        '--opt_path', output_path,
        '--f0method', f0_method
    ]
    #if not (none or empty)
    if index_path:
        cmd.extend([
            '--index_path', index_path,
            '--index_rate', f'{index_rate:.2f}'
        ])

    log_message(f"Process audio: {' '.join(cmd)}\n", log_file=log_file)
    p_out = None
    try:
        p_out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        log_message(f"{PROCESS_FAILED}: {e}", level="ERROR", log_file=log_file)
        return False
    
    if p_out.returncode != 0:
        log_message(f"{PROCESS_FAILED}: {p_out.stdout}\n{p_out.stderr}", level="ERROR", log_file=log_file)
        return False
    else:
        log_message(f"processing audio: {p_out.stdout}", log_file=log_file)

    log_message(f'Process audio completed. from {source_path} to {output_path}', log_file=log_file)
    return True

def process_audio_files(input_dir: str, output_dir: str, model, index_path: str, index_rate: float, splice_size_mb: int):
    #should be auto detected or manually set
    f0up_key = 0
    f0_method = 'rmvpe'
    #value should be distributed based on the available gpu
    gpu = 'cuda:0'
    #should be auto detected based on the gpu
    support_f16 = True
    #today for naming
    today = datetime.datetime.now().strftime('%Y%m%d')
    #on no index path, set index rate to 0
    if not index_path:
        index_rate = 0.0

    has_some_error = False
    file_map = {}

    #any temp data will be stored here
    temp_dir = os.path.join(output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)
    #for each file in the input directory
    file_list = os.listdir(input_dir)
    for i_file, filename in enumerate(file_list):
        log_message(f"Processing file {i_file + 1}/{len(file_list)}: {filename}", log_file=log_file)
        #check if file extension is supported
        is_supported_file = filename.lower().endswith(SUPPORTED_EXTENSION_LIST)
        if not is_supported_file:
            continue
        input_file = os.path.join(input_dir, filename)
        split_dir = os.path.join(temp_dir, os.path.splitext(filename)[0])
        os.makedirs(split_dir, exist_ok=True)

        #check if file size larger than splice size
        file_size = os.path.getsize(input_file) / 1024 / 1024
        enable_splice = file_size > splice_size_mb

        # Split based on silence and limit maximum segment duration
        success = split_audio_on_silence(
            input_file=input_file,
            output_dir=split_dir,
            enable=enable_splice
        )
        if not success:
            log_message(f"Audio splitting failed on File {filename}. SKIP", level="WARNING", log_file=log_file)
            has_some_error = True
            file_map[filename] = SPLIT_FAILED
            continue

        segment_path_list = []
        process_success = False
        #segment is name of segment file
        for segment in sorted(os.listdir(split_dir)):
            segment_path = os.path.join(split_dir, segment)
            output_segment_path = os.path.join(output_dir, f"processed_{segment}")

            process_success = process_audio(segment_path, output_segment_path, model, index_path, index_rate, f0up_key, f0_method, gpu, support_f16)
            if not process_success:
                log_message(f'Processing failed on child file {segment}; parent file {filename}.', level="WARNING", log_file=log_file)
                break
            segment_path_list.append(output_segment_path)
        #end segment if any process failed, discard current file
        if not process_success:
            has_some_error = True
            file_map[filename] = PROCESS_FAILED
            continue
        final_output_path = os.path.join(output_dir, f"rvc{today}-{f0_method}-indexrate{index_rate:.2f}-half{support_f16}_" + filename)
        concat_success = concat_audio_files(segment_path_list, final_output_path)
        if not concat_success:
            has_some_error = True
            log_message(f'Concatenation failed on File {filename}.', level="WARNING", log_file=log_file)
            file_map[filename] = CONCAT_FAILED
            continue
        #store the file map
        file_map[filename] = final_output_path

    shutil.rmtree(temp_dir)
    #remove the processed_ files from output directory
    for filename in os.listdir(output_dir):
        if filename.startswith("processed_"):
            os.remove(os.path.join(output_dir, filename))

    #print file map as {key} => {value}\n
    file_map_str = "".join([f"{k}\n\t=> {v}\n" for k, v in file_map.items()])
    sys.stdout.buffer.writelines([
        '************************************\n'.encode('utf-8'),
        f"File map:\n\n{file_map_str}".encode('utf-8'),
        '************************************\n'.encode('utf-8')
    ])

    return not has_some_error

if __name__ == "__main__":
    #detect if ffmpeg is installed
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, encoding='utf-8')
    except subprocess.CalledProcessError:
        raise "FFmpeg is not installed. Please install FFmpeg first."

    parser = argparse.ArgumentParser(description='Process audio files in batch with splitting.')
    parser.add_argument('--input_dir', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--index_path', help="Path to the index file")
    parser.add_argument('--index_rate', type=float, default=0.6, help="Index rate")
    parser.add_argument('--model', required=True, help="Model name")
    parser.add_argument('--splice_size_mb', type=int, default=0, help="Splice size in MB")
    parser.add_argument('--log_file', default=LOG_FILE, help="Log")
    args = parser.parse_args()
    log_file = args.log_file

    begin_time = datetime.datetime.now()
    log_message(f"BEGIN {begin_time}\n\tinput_dir: {args.input_dir}; output_dir: {args.output_dir}; index_path: {args.index_path}; index_rate: {args.index_rate}; model: {args.model}; splice_size_mb: {args.splice_size_mb}", log_file=log_file)
    all_success = process_audio_files(args.input_dir, args.output_dir, args.model, args.index_path, args.index_rate, args.splice_size_mb)
    end_time = datetime.datetime.now()
    log_message(f"END {begin_time}-{end_time}\n\tTotal time: {end_time - begin_time}", log_file=log_file)
    if not all_success:
        sys.exit(1)
