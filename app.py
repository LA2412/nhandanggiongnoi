# app.py
import pyaudio
import wave
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2ForCTC, Wav2Vec2Processor

# Tải mô hình nhận diện người nói
try:
    speaker_model = Wav2Vec2ForSequenceClassification.from_pretrained("wav2vec2_speaker_id_model")
    speaker_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2_speaker_id_model")
    print("Mô hình nhận diện người nói đã được tải thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình nhận diện người nói: {e}")
    exit()

# Tải mô hình speech-to-text
try:
    stt_model = Wav2Vec2ForCTC.from_pretrained("wav2vec2_vietnamese_speech_to_text_model")
    stt_processor = Wav2Vec2Processor.from_pretrained("wav2vec2_vietnamese_speech_to_text_model")
    print("Mô hình speech-to-text đã được tải thành công!")
except Exception as e:
    print(f"Lỗi khi tải mô hình speech-to-text: {e}")
    exit()

# Định nghĩa nhãn cho 3 người nói
label_map = {0: "Chinh", 1: "Viet", 2: "VietLoi"}

# Hàm ghi âm từ micro
def record_audio(filename, duration=5, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)
    print("Đang ghi âm... (Nhấn Ctrl+C để dừng)")
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Ghi âm xong!")
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Lưu file âm thanh
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

# Hàm nhận diện người nói
def identify_speaker(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze()
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    inputs = speaker_feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = speaker_model(**inputs).logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    return label_map[predicted_id]

# Hàm chuyển âm thanh thành văn bản
def speech_to_text(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.squeeze()
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    inputs = stt_processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = stt_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = stt_processor.batch_decode(predicted_ids)[0]
    return transcription

# Hàm chính để chạy hệ thống
def main():
    audio_file = "recorded_audio.wav"
    try:
        while True:
            # Ghi âm
            record_audio(audio_file, duration=5)
            # Nhận diện người nói
            speaker = identify_speaker(audio_file)
            print(f"Người nói: {speaker}")
            # Chuyển âm thanh thành văn bản
            transcript = speech_to_text(audio_file)
            print(f"Văn bản: {transcript}")
    except KeyboardInterrupt:
        print("Dừng chương trình.")

if __name__ == "__main__":
    main()