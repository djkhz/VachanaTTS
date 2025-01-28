import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from transformers import pipeline
import scipy
from pathlib import Path
import srt
import moviepy.editor as mp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

def get_model_names(model_dir):
    model_paths = Path(model_dir).glob('*')
    return [model_path.name for model_path in model_paths if model_path.is_dir()]

def generate_speech(text, model_dir, model_name, speaking_rate=1.0):
    model_path = os.path.join(model_dir, model_name)
    synthesiser = pipeline("text-to-speech", model=model_path, device=0 if torch.cuda.is_available() else -1)
    speech = synthesiser(text)
    
    # Resample to 48kHz if needed
    if speech["sampling_rate"] != 48000:
        resampled_audio = scipy.signal.resample(speech["audio"][0], int(len(speech["audio"][0]) * 48000 / speech["sampling_rate"]))
        sampling_rate = 48000
    else:
        resampled_audio = speech["audio"][0]
        sampling_rate = speech["sampling_rate"]
    
    return sampling_rate, resampled_audio

def save_audio(sampling_rate, audio_data, filename="output.wav"):
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)
    return filename

def voice_cloning(base_speaker, reference_speaker, model_version, device_choice, vad_select):
    try:
        ckpt_converter = f'./OPENVOICE_MODELS/{model_version}'
        device = "cuda:0" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"
        
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        source_se, _ = se_extractor.get_se(base_speaker, tone_color_converter, vad=vad_select)
        target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=vad_select)
        
        save_path = f'{output_dir}/output_cloned.wav'
        
        tone_color_converter.convert(
            audio_src_path=base_speaker, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
        )
        return save_path, "Voice cloning successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_srt(srt_file):
    with open(srt_file, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))
    return subtitles

def generate_speech_from_srt(srt_file, model_paths, total_duration, speaker_refs):
    subtitles = process_srt(srt_file)
    sampling_rate = 48000
    
    speaker_audio_files = {}
    speaker_timelines = {}
    
    for speaker_id in range(len(model_paths[1])):
        speaker_timelines[speaker_id] = np.zeros(int(total_duration * sampling_rate))
    
    for sub in subtitles:
        text = sub.content.strip()
        if text:
            try:
                if ',' in text:
                    speaker_id_str, text = text.split(',', 1)
                    speaker_id = int(speaker_id_str.strip()) - 1
                else:
                    speaker_id = 0
            except ValueError:
                speaker_id = 0
            
            if 0 <= speaker_id < len(model_paths[1]):
                model_name = model_paths[1][speaker_id]
                
                # Calculate speaking rate to match the timestamped range
                duration = (sub.end - sub.start).total_seconds()
                text_length = len(text.split())
                speaking_rate = text_length / duration
                
                _, audio_data = generate_speech(text, model_paths[0], model_name, speaking_rate=speaking_rate)
                
                start_sample = int(sub.start.total_seconds() * sampling_rate)
                end_sample = int(sub.end.total_seconds() * sampling_rate)
                
                if len(audio_data) > (end_sample - start_sample):
                    audio_data = audio_data[:(end_sample - start_sample)]
                else:
                    audio_data = np.pad(audio_data, (0, end_sample - start_sample - len(audio_data)))
                
                speaker_timelines[speaker_id][start_sample:end_sample] = audio_data
    
    for speaker_id, timeline in speaker_timelines.items():
        if np.any(timeline):
            speaker_file = f"{output_dir}/speaker_{speaker_id + 1}_base.wav"
            save_audio(sampling_rate, timeline, speaker_file)
            speaker_audio_files[speaker_id] = speaker_file
    
    return speaker_audio_files, sampling_rate

def dub_srt(srt_file, media_file, model_dir, model_names, reference_speakers, model_version, device_choice, vad_select, output_type, clone):
    if media_file.endswith(('.mp4', '.mkv', '.avi')):
        media = mp.VideoFileClip(media_file)
    else:
        media = mp.AudioFileClip(media_file)
    total_duration = media.duration
    
    speaker_files, sampling_rate = generate_speech_from_srt(
        srt_file, 
        (model_dir, model_names), 
        total_duration, 
        reference_speakers
    )
    
    if clone:
        cloned_audio_file, status = voice_cloning(
            speaker_files,
            reference_speakers,
            model_version,
            device_choice,
            vad_select
        )
        if cloned_audio_file is None:
            return None, status
        final_audio_file = cloned_audio_file
    else:
        combined_audio = np.zeros(int(total_duration * sampling_rate), dtype=np.float32)
        for speaker_id, file_path in speaker_files.items():
            sr, audio = scipy.io.wavfile.read(file_path)
            combined_audio[:len(audio)] += audio.astype(np.float32) / 32767.0
        combined_audio = np.int16(combined_audio * 32767)
        final_audio_file = f"{output_dir}/final_dubbed.wav"
        scipy.io.wavfile.write(final_audio_file, sampling_rate, combined_audio)
    
    if output_type == "Video" and media_file.endswith(('.mp4', '.mkv', '.avi')):
        final_audio = mp.AudioFileClip(final_audio_file)
        final_video = media.set_audio(final_audio)
        final_video_path = f"{output_dir}/dubbed_video.mp4"
        final_video.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
        return final_video_path, "Dubbing completed!"
    else:
        return final_audio_file, "Dubbing completed!"

if __name__ == "__main__":
    model_dir = "./models_mms"
    model_names = get_model_names(model_dir)
    max_speakers = 4

    srt_file = "path/to/srt_file.srt"
    media_file = "path/to/media_file.mp4"
    model_version = "v2"
    device_choice = "GPU" if torch.cuda.is_available() else "CPU"
    vad_select = True
    output_type = "Video"
    reference_speakers = ["path/to/reference_speaker1.wav", "path/to/reference_speaker2.wav"]
    models = ["model1", "model2"]
    clone = True

    output_file, status = dub_srt(
        srt_file,
        media_file,
        model_dir,
        models,
        reference_speakers,
        model_version,
        device_choice,
        vad_select,
        output_type,
        clone
    )
    print(f"Output file: {output_file}, Status: {status}")
