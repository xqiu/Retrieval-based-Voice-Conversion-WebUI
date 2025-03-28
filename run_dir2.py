import os
import subprocess
import argparse
import sys
import shutil
import re

# (Optional) Duration constraints—you may use them later if needed.
MIN_SEGMENT_DURATION = 120  # 2 minutes
MAX_SEGMENT_DURATION = 180  # 3 minutes

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'segment_log.txt')
def log_message(message):
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

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
        # 5.1 if 6
        # 7.1 if 8
        channel_str = 'mono' if channel == '1' else 'stereo' if channel == '2' else '5.1' if channel == '6' else '7.1' if channel == '8' else 'unknown'

        log_message(f"get_audio_detail {input_file} => \n\tduration: {duration}, \n\tsample_rate: {sample_rate}, \n\tchannel: {channel}")

        return float(duration), int(sample_rate), channel_str
    except (subprocess.CalledProcessError, ValueError) as e:
        return -1.0, -1, 'unknown'

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
    log_message(f'append silence to {input_file} by {append_duration} seconds')
    # output file name = input file name appended with '_append'
    output_file = input_file
    if '.wav' in input_file:
        output_file = input_file.rsplit('.wav', 1)[0] + '_append.wav'
        
    #if output_file already exists, skip it
    if continue_job and os.path.exists(output_file):
        print(f"[WARNING] Skipping appending silence to {input_file} because {output_file} already exists")
        return output_file
    
    ffmpeg_silence_cmd = [
        'ffmpeg', '-y', '-i', input_file, '-f', 'lavfi', '-t', str(append_duration), '-i', f'anullsrc=channel_layout={channel}:sample_rate={sample_rate}', '-filter_complex', '[0][1]concat=n=2:v=0:a=1', output_file
    ]
    try: 
        subprocess.run(ffmpeg_silence_cmd, check=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        log_message(f"[Error] appending silence: {e}")
        return input_file

    return output_file

def split_audio_into_segments(input_file, temp_dir, noise_threshold=-30, silence_duration=1, continue_job=False):
    """
    Splits the input audio file into segments labeled as 'silence' and 'non_silence'
    using ffmpeg's silencedetect filter.
    
    Returns a list of tuples: (segment_file, segment_type, start_time, end_time)
    """
    # Run ffmpeg silencedetect to get silence intervals.
    silence_cmd = [
        'ffmpeg', '-i', input_file, '-af',
        f'silencedetect=noise={noise_threshold}dB:d={silence_duration}', '-f', 'null', '-'
    ]

    result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8')
    
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

    # Get total duration of the input file.
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    total_duration = float(subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True, check=True).stdout.strip())
    log_message(f'BEGIN file => {input_file}')
    log_message(f'\t=> total_duration: {total_duration}')
    # Create segments: non-silence segments are between silence intervals.
    segments = []  # Each element is (start_time, end_time, segment_type)
    current_time = 0.0
    log_message(f'BEGIN RAW silence_intervals')
    for (silence_start, silence_end) in silence_intervals:
        log_message(f'{silence_start}\t{silence_end}')
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
        if adjust_silence_start > current_time:
            segments.append((current_time, adjust_silence_start, 'non_silence'))
        # Add the silence segment.
        segments.append((adjust_silence_start, adjust_silence_end, 'silence'))
        current_time = adjust_silence_end
    log_message(f'END RAW silence_intervals')
    # If there is audio after the last silence, add it.
    if current_time < total_duration:
        segments.append((current_time, total_duration, 'non_silence'))

    # Extract each segment to its own file.
    extracted_segments = []
    log_message(f'BEGIN segments')
    for idx, (start, end, seg_type) in enumerate(segments):
        log_message(f'{start}\t{end}\t{seg_type}')
        duration = end - start
        if(duration < 0.05):
            print(f"[WARNING] Skipping short segment of file {input_file}:\n\t {start}-{end} ({duration}s)")
            log_message(f"[WARNING] Skipping short segment of file {input_file}:\n\t {start}-{end} ({duration}s)")
            continue
        segment_filename = os.path.join(temp_dir, f"{idx:03d}_{seg_type}.wav")

        #if segment_filename already exists, skip it
        if continue_job and os.path.exists(segment_filename):
            extracted_segments.append((segment_filename, seg_type, start, end))
            print(f"[WARNING] Skipping segment {segment_filename} because it already exists")
            continue
        ffmpeg_extract_cmd = [
            'ffmpeg', '-y', '-i', input_file, '-ss', str(start), '-t', str(duration),
            '-c', 'copy', segment_filename
        ]
        subprocess.run(ffmpeg_extract_cmd, check=True, encoding='utf-8')
        extracted_segments.append((segment_filename, seg_type, start, end))
    log_message(f'END segments')
    
    return extracted_segments

def batch_concat(segments, batch_size, temp_dir, final_output, continue_job):
    if continue_job and os.path.exists(final_output):
        print(f"[WARNING] Skipping batch concat because {final_output} already exists")
        return

    import math
    num_batches = math.ceil(len(segments) / batch_size)
    intermediate_files = []
    
    for batch in range(num_batches):
        batch_segments = segments[batch*batch_size : (batch+1)*batch_size]
        inputs = []
        for seg in batch_segments:
            inputs.extend(['-i', seg])
        filter_complex = "".join([f"[{i}:0]" for i in range(len(batch_segments))])
        filter_complex += f"concat=n={len(batch_segments)}:v=0:a=1[out]"
        intermediate_output = os.path.join(temp_dir, f"intermediate_{batch:03d}.wav")
        intermediate_files.append(intermediate_output)
        
        cmd = [
            'ffmpeg', '-y', *inputs,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:a', 'pcm_s16le', intermediate_output
        ]
        print("Running batch concat command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, encoding='utf-8')
    
    # Now merge all intermediate files into the final output.
    inputs = []
    for file in intermediate_files:
        inputs.extend(['-i', file])
    filter_complex = "".join([f"[{i}:0]" for i in range(len(intermediate_files))])
    filter_complex += f"concat=n={len(intermediate_files)}:v=0:a=1[out]"
    final_cmd = [
        'ffmpeg', '-y', *inputs,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:a', 'pcm_s16le', final_output
    ]
    print("Running final concat command:")
    print(" ".join(final_cmd))
    subprocess.run(final_cmd, check=True, encoding='utf-8')
    
    # Optionally, clean up intermediate files.
    for file in intermediate_files:
        os.remove(file)

def process_audio_files(input_dir, output_dir, model, index, continue_job=False):
    supported_extensions = ('.wav', '.mp3')
    venv_python = sys.executable

    # Create a temporary directory to hold segments.
    temp_dir = os.path.join(output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each file in the input directory.
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            input_file = os.path.join(input_dir, filename)
            base_name, _ = os.path.splitext(filename)
            
            # Create a subdirectory for segments from this file.
            file_temp_dir = os.path.join(temp_dir, base_name)
            os.makedirs(file_temp_dir, exist_ok=True)
            
            # Split the file into silence and non-silence segments.
            segments = split_audio_into_segments(
                input_file, file_temp_dir, noise_threshold=-30, silence_duration=1, continue_job=continue_job
            )
            
            # Process each segment accordingly.
            final_segments = []  # To hold tuples (processed_segment_path, original_start_time)
            for seg_file, seg_type, start, end in segments:
                processed_seg = os.path.join(file_temp_dir, f"processed_{os.path.basename(seg_file)}")
                #if target already exists, skip it
                if continue_job and os.path.exists(processed_seg):
                    print (f"[WARNING] Skipping processed segment {processed_seg} because it already exists")
                else:
                    if seg_type == 'non_silence':
                        # Process non-silence segments using infer_cli.py.
                        cmd = [
                            venv_python, 'tools/infer_cli.py',
                            '--f0up_key', '0',
                            '--input_path', seg_file,
                            '--index_path', index,
                            '--model_name', model,
                            '--index_rate', '0.6',
                            '--device', 'cuda:0',
                            '--is_half', 'True',
                            '--opt_path', processed_seg,
                            '--f0method', 'rmvpe'
                        ]
                        print('################################')
                        print(f"Processing non-silence segment: {seg_file}")
                        print(f"Running command: {' '.join(cmd)}")
                        print('################################')
                        subprocess.run(cmd, check=True, encoding='utf-8')
                    else:
                        # For silence segments, simply copy the file (do not process).
                        shutil.copy(seg_file, processed_seg)

                # after processing, compare duration of seg_file and processed_seg
                # if processed_seg is shorter than seg_file, append silence to the end of processed_seg
                (seg_duration, seg_sample_rate, seg_channel) = get_audio_detail(seg_file)
                (processed_seg_duration, processed_seg_sample_rate, processed_seg_channel) = get_audio_detail(processed_seg)
                log_message(f"seg_file: {seg_file}\td: {seg_duration}s\tsr: {seg_sample_rate} Hz\tch: {seg_channel}")
                log_message(f"processed_seg: {processed_seg}\td: {processed_seg_duration}\tsr: {processed_seg_sample_rate}\tch: {processed_seg_channel}")
                if processed_seg_duration < seg_duration:
                    append_duration = seg_duration - processed_seg_duration
                    processed_seg = append_silence(processed_seg, append_duration, processed_seg_sample_rate, processed_seg_channel, continue_job=continue_job)
                
                final_segments.append((processed_seg, start))
            
            # Reorder segments by their original start times.
            final_segments_sorted = sorted(final_segments, key=lambda x: x[1])
            
            # Write a concat file for ffmpeg.
            concat_list = os.path.join(file_temp_dir, "concat.txt")
            with open(concat_list, 'w') as f:
                for seg, _ in final_segments_sorted:
                    f.write(f"file '{seg}'\n")
            
            final_output_path = os.path.join(output_dir, f"rvc20241205-rmvpe-indexrate0.6-halftrue__{base_name}.wav")
            
            # Build a list of inputs for ffmpeg based on your sorted processed segments.
            # inputs = []
            # for seg, _ in final_segments_sorted:
            #     inputs.extend(['-i', seg])
            # num_segments = len(final_segments_sorted)

            # # Build the filter complex string.
            # # This creates a string like: "[0:0][1:0]concat=n=2:v=0:a=1[out]"
            # filter_complex = "".join([f"[{i}:0]" for i in range(num_segments)])
            # filter_complex += f"concat=n={num_segments}:v=0:a=1[out]"

            # # Build the ffmpeg command using the concat filter.
            # concat_cmd = [
            #     'ffmpeg', *inputs,
            #     '-filter_complex', filter_complex,
            #     '-map', '[out]',
            #     '-c:a', 'pcm_s16le',  # Re-encode to ensure timestamps are regenerated.
            #     final_output_path
            # ]
            # print("Running concat filter command:")
            # print(" ".join(concat_cmd))
            # subprocess.run(concat_cmd, check=True)


            all_segments = [seg for seg, _ in final_segments_sorted]
            batch_concat(all_segments, batch_size=100, temp_dir=file_temp_dir, final_output=final_output_path, continue_job=continue_job)

            print(f"Merged file created at: {final_output_path}")
            
    # Clean up the global temporary segments directory.
    # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files by splitting into silence and non-silence segments.')
    parser.add_argument('--input_dir', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--index_path', help="Path to the index file")
    parser.add_argument('--model', required=True, help="Name of the model to use")
    parser.add_argument('--continue_job', action='store_true', help="Continue doing the job if intermediate files already exist")
    args = parser.parse_args()

    process_audio_files(args.input_dir, args.output_dir, args.model, args.index_path, args.continue_job)
