import datetime
import os
import subprocess
import argparse
import sys
import shutil
import re

RVC_PATH = os.path.dirname(os.path.realpath(__file__))
#python path is assigned by conda
VENV_PATH = sys.executable
#this is default log file
LOG_FILE = os.path.join(RVC_PATH, "run_infer.log")
log_file = None
#BEGIN ERROR CODE
SPLIT_FAILED = "SPLIT_FAILED"
PROCESS_FAILED = "PROCESS_FAILED"
CONCAT_FAILED = "CONCAT_FAILED"
#END ERROR CODE

def log_message(message, level="INFO", log_file=LOG_FILE):
    if log_file is LOG_FILE:
        print(f"[WARNING] using default log file => {message}")
    with open(log_file, "a", encoding='utf-8') as log_file:
        log_file.write(f"[{level}] {message}\n")

#BEGIN CONSTANT

# Minimum segment duration in seconds
DEFAULT_MIN_SEGMENT_DURATION = 120
# Maximum segment duration in seconds
DEFAULT_MAX_SEGMENT_DURATION = 180
# Noise threshold in dB
DEFAULT_NOISE_THRESHOLD = -50
# Minimum silence duration in seconds
DEFAULT_SILENCE_DURATION = 0.5

DEFAULT_CONCAT_BATCH_SIZE = 100

SEGMENT_SILENCE = 'silence'
SEGMENT_NON_SILENCE = 'non_silence'

SUPPORTED_EXTENSION_LIST = ('.wav', '.mp3')

#END CONSTANT

def get_audio_detail(input_file):
    """_summary_

    Args:
        input_file (_str_): path of audio file (wav or mp3)

    Returns:
        tuple: (duration: float, sample_rate: int, channel_str: str)
        duration: unit second
        sample_rate: unit Hz
        channel_str: mono, stereo, 5.1, 7.1, unknown
    """
    #using ffprobe to get the duration of the audio file
    detail_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 
        # ffprobe use it's own order. even you write duration,sample_rate,channels, it still return data in the order of sample_rate, channels, duration
        'stream=sample_rate,channels,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        output = subprocess.run(detail_cmd, stdout=subprocess.PIPE, text=True, check=True, encoding='utf-8').stdout.strip()
        sample_rate, channel, duration = output.split('\n')

        #set channel str
        # mono if 1
        # stereo if 2
        # 5.1 if 16
        # 7.1 if 8
        channel_str = 'mono' if channel == '1' else 'stereo' if channel == '2' else '5.1' if channel == '6' else '7.1' if channel == '8' else 'unknown'
        return float(duration), int(sample_rate), channel_str
    except (subprocess.CalledProcessError, ValueError) as e:
        return -1.0, -1, 'unknown'

def get_total_duration(input_file: str):
    global log_file
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        return float(subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True, check=True, encoding='utf-8').stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        log_message(f"{SPLIT_FAILED} getting audio duration: {e}", level="ERROR", log_file=log_file)
        return None

def get_silient_times(input_file: str, noise_threshold: int, silence_duration: float):
    """
    Detects silence intervals in the input audio file using ffmpeg's silencedetect filter.
    Return: 
    - List of tuples (silence_start, silence_end) representing the start and end times of each silence interval.
    - None if an error occurred.
    """
    global log_file

    silence_cmd = [
        'ffmpeg', '-i', input_file, '-af',
        f'silencedetect=noise={noise_threshold}dB:d={silence_duration}', '-f', 'null', '-'
    ]
    log_message(f"Detecting silence in {input_file} with noise threshold {noise_threshold} dB and silence duration {silence_duration} seconds", log_file=log_file)
    try:
        result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8')
    except Exception as e:
        log_message(f"{SPLIT_FAILED} detecting silence: {e}", level="ERROR", log_file=log_file)
        return None

    # Parse the stderr output to get pairs of silence_start and silence_end.
    silence_intervals = []
    current_silence_start = None
    for line in result.stderr.splitlines():
        if 'silence_start' in line:
            match = re.search(r'silence_start: (\d+\.?\d*)', line)
            if match:
                current_silence_start = float(match.group(1))
        if 'silence_end' in line and current_silence_start is not None:
            match = re.search(r'silence_end: (\d+\.?\d*)', line)
            if match:
                silence_end = float(match.group(1))
                silence_intervals.append((current_silence_start, silence_end))
                current_silence_start = None
    #log_message(f"Silence intervals: {silence_intervals}", log_file=log_file)
    return silence_intervals

def append_silence(input_file: str, append_duration: float, sample_rate: int, channel: str, continue_job) -> bool:
    """
    Append silence to the end of the input audio file.

    Args:
        input_file (str): input audio file path
        append_duration (float): duration of silence to append in seconds
        sample_rate (int): sample rate of the audio file
        channel (str): channel of the audio file
    Returns:
        str: output file path (if fail will return same path as input_file)
    """
    # output file name = input file name appended with '_append'
    output_file = input_file
    if input_file.endswith('.wav'):
        output_file = input_file.rsplit('.wav', 1)[0] + '_append.wav'
    #if output_file already exists, skip it
    if continue_job and os.path.exists(output_file):
        log_message(f"Skipping appending silence to {input_file} because {output_file} already exists", log_file=log_file, level='WARNING')
        return output_file
    
    log_message(f'append silence to {input_file} by {append_duration} seconds', log_file=log_file)
    ffmpeg_silence_cmd = [
        'ffmpeg', '-y', '-i', input_file, '-f', 'lavfi', '-t', str(append_duration), '-i', f'anullsrc=channel_layout={channel}:sample_rate={sample_rate}', '-filter_complex', '[0][1]concat=n=2:v=0:a=1', output_file
    ]
    try: 
        subprocess.run(ffmpeg_silence_cmd, check=True, encoding='utf-8')
    except Exception as e:
        log_message(f"appending silence: {e}", level="ERROR", log_file=log_file)
        return input_file

    return output_file

def split_audio_into_segments(input_file: str, output_dir: str, min_duration: int, max_duration: int, noise_threshold: int, silence_duration: float, enable: bool, continue_job: bool):
    """
    Splits the input audio file into segments labeled as 'silence' and 'non_silence'
    using ffmpeg's silencedetect filter.

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
    silence_intervals = get_silient_times(input_file, noise_threshold, silence_duration) if enable else []
    if silence_intervals is None:
        #since file too large will cause problem, give up this job
        #log_message('Error: Failed to detect silence intervals.', log_file=log_file)
        return False
    
    # Get the total duration of the audio
    total_duration = get_total_duration(input_file)
    if total_duration is None:
        #log_message('Error: Failed to get the total duration of the audio.', log_file=log_file)
        return False
    
    log_message(f'BEGIN file => {input_file}', log_file=log_file)
    log_message(f'\t=> total_duration: {total_duration}', log_file=log_file)
    
    # Create segments: non-silence segments are between silence intervals.
    segments = []  # Each element is (start_time, end_time, segment_type)
    current_time = 0.0
    for (silence_start, silence_end) in silence_intervals:
        adjust_silence_start = silence_start
        adjust_silence_end = silence_end
        #adjust silence_start and silence_end
        # 1. precision only to 1 decimal place
        # 2. adjust_silence_start will be greater or equal to silence_start
        # 3. adjust_silence_end will be less or equal to silence_end
        # adjust_silence_start = round(silence_start, 1)
        # adjust_silence_end = round(silence_end, 1)
        # if adjust_silence_start < silence_start:
        #     adjust_silence_start += 0.1
        # if adjust_silence_end > silence_end:
        #     adjust_silence_end -= 0.1
        # If there is audio before this silence, add it as a non-silence segment.

        # If there is audio before this silence, add it as a non-silence segment.
        if adjust_silence_start > current_time:
            segments.append((current_time, adjust_silence_start, SEGMENT_NON_SILENCE))
        # Add the silence segment.
        segments.append((adjust_silence_start, adjust_silence_end, SEGMENT_SILENCE))
        current_time = adjust_silence_end

    # If there is audio after the last silence, add it.
    if current_time < total_duration:
        segments.append((current_time, total_duration, SEGMENT_NON_SILENCE))

    # Extract each segment to its own file.
    extracted_segments = []
    log_message(f'BEGIN segments', log_file=log_file)
    for idx, (start, end, seg_type) in enumerate(segments):
        log_message(f'{start}\t{end}\t{seg_type}', log_file=log_file)
        duration = end - start
        if(duration < 0.05):
            log_message(f"Skipping short segment of file {input_file}:\n\t {start}-{end} ({duration}s)", level='WARNING', log_file=log_file)
            continue
        segment_filename = os.path.join(output_dir, f"{idx:03d}_{seg_type}.wav")

        #if segment_filename already exists, skip it
        if continue_job and os.path.exists(segment_filename):
            extracted_segments.append((segment_filename, seg_type, start, end))
            log_message(f'Skipping segment {segment_filename} because it already exists', level='WARNING', log_file=log_file)
            continue
        ffmpeg_extract_cmd = [
            'ffmpeg', '-y', '-i', input_file, '-ss', str(start), '-t', str(duration),
            '-c', 'copy', segment_filename
        ]
        try:
            subprocess.run(ffmpeg_extract_cmd, check=True, encoding='utf-8')
        except Exception as e:
            #since we cannot process every segment, give up this job
            log_message(f"{SPLIT_FAILED} extracting segment: {e}", level="ERROR", log_file=log_file)
            return []
        extracted_segments.append((segment_filename, seg_type, start, end))
    log_message(f'END segments', log_file=log_file)
    return extracted_segments

def sub_concat_audio(sub_segment_list, output_filepath):
    global log_file
    inputs = []
    for seg in sub_segment_list:
        inputs.extend(['-i', seg])
    filter_complex = "".join([f"[{i}:0]" for i in range(len(sub_segment_list))])
    filter_complex += f"concat=n={len(sub_segment_list)}:v=0:a=1[out]"
    intermediate_output = output_filepath
    
    cmd = [
        'ffmpeg', '-y', *inputs,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:a', 'pcm_s16le', intermediate_output
    ]
    log_message(f"Running batch concat command: {cmd}", log_file=log_file)
    try:
        subprocess.run(cmd, check=True, encoding='utf-8')
    except Exception as e:
        log_message(f"{CONCAT_FAILED}: {e}", level="ERROR", log_file=log_file)
        return False
    return True

def concat_audio_files(segment_pair_list, output_path: str, temp_dir: str,batch_size: int, continue_job: bool):
    """
    Concatenate multiple audio files into a single file.

    Parameters:
    - segment_pair_list (list(segment_file_path, start_time)): List of input audio files to concatenate.
    - output_path (str): Path to save the concatenated audio file.
    - temp_dir (str): Directory to store temporary files.
    - batch_size (int): Number of files to concatenate in each batch which prevent ffmpeg from running out of memory.
    - continue_job (bool): Continue the job if the output file already exists.

    Returns:
    - bool: True if concatenation was successful, False otherwise.
    """
    global log_file

    if continue_job and os.path.exists(output_path):
        log_message(f"Skipping batch concat because {output_path} already exists", level="WARNING", log_file=log_file)
        return True
    
    import math
    concat_list_path = os.path.join(temp_dir, f'concat.txt')
    segments = []
    with open(concat_list_path, 'w', encoding='utf-8') as f:
        for input_file, _ in segment_pair_list:
            segments.append(input_file)
            f.write(f"file '{input_file}'\n")

    num_batches = math.ceil(len(segments) / batch_size)
    intermediate_files = []

    for batch in range(num_batches):
        batch_segments = segments[batch*batch_size : (batch+1)*batch_size]
        intermediate_output = os.path.join(temp_dir, f'intermediate_{batch:03d}.wav')
        success = sub_concat_audio(batch_segments, intermediate_output)
        if not success:
            return False
        intermediate_files.append(intermediate_output)

    # Concatenate the intermediate files.
    success = sub_concat_audio(intermediate_files, output_path)
    if not success:
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
    except Exception as e:
        log_message(f"{PROCESS_FAILED}: {e}", level="ERROR", log_file=log_file)
        return False
    
    if p_out.returncode != 0:
        log_message(f"{PROCESS_FAILED}: {p_out.stdout}\n{p_out.stderr}", level="ERROR", log_file=log_file)
        return False
    else:
        log_message(f"processing audio: {p_out.stdout}", log_file=log_file)

    log_message(f'Process audio completed. from {source_path} to {output_path}', log_file=log_file)
    return True

def process_audio_files(input_dir: str, output_dir: str, model, index_path: str, index_rate: float, splice_size_mb: int, min_segment_duration: int, max_segment_duration: int, noise_threshold: int, silence_duration: float, continue_job: bool):
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
        base_name, _ = os.path.splitext(filename)
        # Create a subdirectory for segments from this file.
        file_temp_dir = os.path.join(temp_dir, base_name)
        os.makedirs(file_temp_dir, exist_ok=True)

        #check if file size larger than splice size
        file_size_md = os.path.getsize(input_file) / 1024 / 1024
        enable_splice = file_size_md > splice_size_mb

        # Split based on silence and limit maximum segment duration
        segments = split_audio_into_segments(
            input_file=input_file,
            output_dir=file_temp_dir,
            min_duration=min_segment_duration,
            max_duration=max_segment_duration,
            noise_threshold=noise_threshold,
            silence_duration=silence_duration,
            enable=enable_splice,
            continue_job=continue_job
        )
        if not segments:
            log_message(f"Audio splitting failed on File {filename}. SKIP", level="WARNING", log_file=log_file)
            has_some_error = True
            file_map[filename] = SPLIT_FAILED
            continue

        # Process each segment accordingly.
        final_segments = []  # To hold tuples 
        process_success = False
        #segment is name of segment file
        for seg_file, seg_type, start, end in segments:
            processed_seg_file = os.path.join(file_temp_dir, f"processed_{os.path.basename(seg_file)}")
            #if target already exists, skip it
            if continue_job and os.path.exists(processed_seg_file):
                log_message(f"Skipping processed segment {processed_seg_file} because it already exists", level="WARNING", log_file=log_file)
            else:
                if seg_type == SEGMENT_NON_SILENCE:
                    process_success = process_audio(seg_file, processed_seg_file, model, index_path, index_rate, f0up_key, f0_method, gpu, support_f16)
                    if not process_success:
                        log_message(f'Processing failed on child file {seg_file}; parent file {filename}.', level="WARNING", log_file=log_file)
                        break
                else:
                    # Copy silence segments as they are.
                    shutil.copy(seg_file, processed_seg_file)
            
            # after processing, compare duration of seg_file and processed_seg
            # if processed_seg is shorter than seg_file, append silence to the end of processed_seg
            (seg_duration, seg_sample_rate, seg_channel) = get_audio_detail(seg_file)
            (processed_seg_duration, processed_seg_sample_rate, processed_seg_channel) = get_audio_detail(processed_seg_file)
            log_message(f"\tseg_file: {seg_file}\n\t\td: {seg_duration}s\n\t\tsr: {seg_sample_rate} Hz\n\t\tch: {seg_channel}", log_file=log_file)
            log_message(f"\tprocessed_seg: {processed_seg_file}\n\t\td: {processed_seg_duration}\n\t\tsr: {processed_seg_sample_rate}\n\t\tch: {processed_seg_channel}", log_file=log_file)
            if processed_seg_duration < seg_duration:
                append_duration = seg_duration - processed_seg_duration
                processed_seg_file = append_silence(processed_seg_file, append_duration, processed_seg_sample_rate, processed_seg_channel, continue_job=continue_job)

            final_segments.append((processed_seg_file, start))

        #end segment if any process failed, discard current file
        if not process_success:
            has_some_error = True
            file_map[filename] = PROCESS_FAILED
            continue
        # Reorder segments by their original start times.
        final_segments_sorted = sorted(final_segments, key=lambda x: x[1])

        final_output_path = os.path.join(output_dir, f"rvc{today}-{f0_method}-indexrate{index_rate:.2f}-half{support_f16}_{base_name}.wav")
        concat_success = concat_audio_files(final_segments_sorted, final_output_path, temp_dir=file_temp_dir,batch_size=DEFAULT_CONCAT_BATCH_SIZE, continue_job=continue_job)
        if not concat_success:
            has_some_error = True
            log_message(f'Concatenation failed on File {filename}.', level="WARNING", log_file=log_file)
            file_map[filename] = CONCAT_FAILED
            continue
        #store the file map
        file_map[filename] = final_output_path

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
    except:
        sys.stdout.buffer.writelines([
            "FFmpeg is not installed. Please install FFmpeg first.".encode('utf-8')
        ])
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Process audio files in batch with splitting.')
    parser.add_argument('--input_dir', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--index_path', help="Path to the index file")
    parser.add_argument('--index_rate', type=float, default=0.6, help="Index rate")
    parser.add_argument('--model', required=True, help="Model name")
    parser.add_argument('--splice_size_mb', type=int, default=0, help="Splice size in MB")
    parser.add_argument('--log_file', default=LOG_FILE, help="Log")
    parser.add_argument('--min_segment_duration', type=int, default=DEFAULT_MIN_SEGMENT_DURATION, help="Minimum segment duration in seconds")
    parser.add_argument('--max_segment_duration', type=int, default=DEFAULT_MAX_SEGMENT_DURATION, help="Maximum segment duration in seconds")
    parser.add_argument('--noise_threshold', type=int, default=DEFAULT_NOISE_THRESHOLD, help="Noise threshold in dB for silence detection")
    parser.add_argument('--silence_duration', type=float, default=DEFAULT_SILENCE_DURATION, help="Minimum silence duration in seconds")
    parser.add_argument('--continue_job', action='store_true', help="Continue the job if the output file already exists; on not set, will overwrite the existing file")

    args = parser.parse_args()
    log_file = args.log_file

    begin_time = datetime.datetime.now()
    log_message(f"BEGIN {begin_time}\n\tinput_dir: {args.input_dir}; output_dir: {args.output_dir}; index_path: {args.index_path}; "
    f"index_rate: {args.index_rate}; model: {args.model}; splice_size_mb: {args.splice_size_mb}; "
    f"min_segment_duration: {args.min_segment_duration}; max_segment_duration: {args.max_segment_duration}; "
    f"noise_threshold: {args.noise_threshold}; silence_duration: {args.silence_duration}", log_file=log_file)

    all_success = process_audio_files(args.input_dir, args.output_dir, args.model, args.index_path, args.index_rate, args.splice_size_mb,
    args.min_segment_duration, args.max_segment_duration, args.noise_threshold, args.silence_duration, args.continue_job)

    end_time = datetime.datetime.now()
    log_message(f"END {begin_time}-{end_time}\n\tTotal time: {end_time - begin_time}", log_file=log_file)
    if not all_success:
        sys.exit(1)
