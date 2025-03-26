import os
import subprocess
import argparse
import sys
import tempfile
import shutil
import re

# (Optional) Duration constraints—you may use them later if needed.
MIN_SEGMENT_DURATION = 120  # 2 minutes
MAX_SEGMENT_DURATION = 180  # 3 minutes

def split_audio_into_segments(input_file, temp_dir, noise_threshold=-30, silence_duration=1):
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

    result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, text=True, check=True)
    
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

    # Create segments: non-silence segments are between silence intervals.
    segments = []  # Each element is (start_time, end_time, segment_type)
    current_time = 0.0
    for (silence_start, silence_end) in silence_intervals:
        # If there is audio before this silence, add it as a non-silence segment.
        if silence_start > current_time:
            segments.append((current_time, silence_start, 'non_silence'))
        # Add the silence segment.
        segments.append((silence_start, silence_end, 'silence'))
        current_time = silence_end
    # If there is audio after the last silence, add it.
    if current_time < total_duration:
        segments.append((current_time, total_duration, 'non_silence'))

    # Extract each segment to its own file.
    extracted_segments = []
    for idx, (start, end, seg_type) in enumerate(segments):
        duration = end - start
        if(duration < 0.05):
            print(f"Skipping short segment: {start}-{end} ({duration}s)")
            continue
        segment_filename = os.path.join(temp_dir, f"{idx:03d}_{seg_type}.wav")
        ffmpeg_extract_cmd = [
            'ffmpeg', '-i', input_file, '-ss', str(start), '-t', str(duration),
            '-c', 'copy', segment_filename
        ]
        subprocess.run(ffmpeg_extract_cmd, check=True)
        extracted_segments.append((segment_filename, seg_type, start, end))
    
    return extracted_segments

def batch_concat(segments, batch_size, temp_dir, final_output):
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
            'ffmpeg', *inputs,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-c:a', 'pcm_s16le', intermediate_output
        ]
        print("Running batch concat command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    
    # Now merge all intermediate files into the final output.
    inputs = []
    for file in intermediate_files:
        inputs.extend(['-i', file])
    filter_complex = "".join([f"[{i}:0]" for i in range(len(intermediate_files))])
    filter_complex += f"concat=n={len(intermediate_files)}:v=0:a=1[out]"
    final_cmd = [
        'ffmpeg', *inputs,
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:a', 'pcm_s16le', final_output
    ]
    print("Running final concat command:")
    print(" ".join(final_cmd))
    subprocess.run(final_cmd, check=True)
    
    # Optionally, clean up intermediate files.
    for file in intermediate_files:
        os.remove(file)


def process_audio_files(input_dir, output_dir):
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
                input_file, file_temp_dir, noise_threshold=-30, silence_duration=1
            )
            
            # Process each segment accordingly.
            final_segments = []  # To hold tuples (processed_segment_path, original_start_time)
            for seg_file, seg_type, start, end in segments:
                processed_seg = os.path.join(file_temp_dir, f"processed_{os.path.basename(seg_file)}")
                if seg_type == 'non_silence':
                    # Process non-silence segments using infer_cli.py.
                    cmd = [
                        venv_python, 'tools/infer_cli.py',
                        '--f0up_key', '0',
                        '--input_path', seg_file,
                        '--index_path', 'C:\\AI\\Retrieval-based-Voice-Conversion-WebUI\\logs\\jinbodhi3\\trained_IVF2963_Flat_nprobe_1_jinbodhi1_v2.index',
                        '--model_name', 'jinbodhi3.pth',
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
                    subprocess.run(cmd, check=True)
                else:
                    # For silence segments, simply copy the file (do not process).
                    shutil.copy(seg_file, processed_seg)
                
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
            batch_concat(all_segments, batch_size=100, temp_dir=file_temp_dir, final_output=final_output_path)

            print(f"Merged file created at: {final_output_path}")
            
    # Clean up the global temporary segments directory.
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files by splitting into silence and non-silence segments.')
    parser.add_argument('--input_dir', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    args = parser.parse_args()

    process_audio_files(args.input_dir, args.output_dir)
