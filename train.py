# train.py
import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)

# Hàm fine-tune cho Speaker Identification
def train_speaker_identification():
    # Định nghĩa nhãn cho 3 người nói
    label_map = {"chinh": 0, "viet": 1, "VietLoi": 2}

    # Chuẩn bị dữ liệu
    dataset_path = "D:/nhandanggiongnoi/samples"
    data = []
    for speaker, label in label_map.items():
        speaker_path = os.path.join(dataset_path, speaker)
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                # Đảm bảo label là int trước khi đưa vào Dataset
                data.append({"path": file_path, "label": int(label)})

    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Tải feature extractor từ mô hình cục bộ
    model_name = "D:/nhandanggiongnoi/wav2vec2-large-xlsr-53"
    print("Tải feature extractor cho speaker identification...")
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        print("Feature extractor đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải feature extractor: {e}")
        return

    # Hàm tiền xử lý dữ liệu
    def preprocess_function(batch):
        # Chuyển đổi dữ liệu âm thanh thành mảng NumPy
        audio_array = batch["path"]["array"]
        # Đảm bảo audio_array là mảng NumPy
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        # Chuẩn hóa sampling rate và padding
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160000
        )
        batch["input_values"] = inputs.input_values[0]
        # Chuyển đổi nhãn sang torch.long
        label = batch["label"]
        if isinstance(label, torch.Tensor):
            batch["label"] = label.to(dtype=torch.long)
        else:
            batch["label"] = torch.tensor(label, dtype=torch.long)
        return batch

    dataset = dataset.map(preprocess_function, remove_columns=["path"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Kiểm tra và chuyển đổi kiểu dữ liệu của nhãn
    def ensure_label_dtype(example):
        example["label"] = torch.tensor(example["label"], dtype=torch.long)
        return example

    train_dataset = train_dataset.map(ensure_label_dtype)
    test_dataset = test_dataset.map(ensure_label_dtype)

    # Tải mô hình để fine-tune
    print("Tải mô hình để fine-tune speaker identification...")
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name, num_labels=len(label_map), problem_type="single_label_classification"
        )
        print("Mô hình đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Cấu hình huấn luyện
    training_args = TrainingArguments(
        output_dir="./wav2vec2_speaker_id",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True
    )

    # Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor
    )

    # Huấn luyện và lưu mô hình
    print("Bắt đầu huấn luyện mô hình speaker identification...")
    trainer.train()
    trainer.evaluate()
    trainer.save_model("wav2vec2_speaker_id_model")
    feature_extractor.save_pretrained("wav2vec2_speaker_id_model")
    print("Huấn luyện speaker identification hoàn tất!")

# Hàm fine-tune cho Speech-to-Text
def train_speech_to_text():
    # Chuẩn bị dữ liệu (bao gồm transcript)
    dataset_path = "D:/nhandanggiongnoi/samples"
    speakers = ["chinh", "viet", "VietLoi"]
    data = []
    for speaker in speakers:
        speaker_path = os.path.join(dataset_path, speaker)
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                transcript_file = file.replace(".wav", ".txt")
                transcript_path = os.path.join(speaker_path, transcript_file)
                if os.path.exists(transcript_path):
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                    data.append({"path": file_path, "transcript": transcript})

    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # Tải mô hình tiếng Việt
    model_name = "vinai/vietnamese-wav2vec2-large"
    print("Tải tokenizer và feature extractor cho speech-to-text...")
    try:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        print("Processor đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải: {e}")
        return

    # Hàm tiền xử lý dữ liệu
    def preprocess_function(batch):
        # Chuyển đổi dữ liệu âm thanh thành mảng NumPy
        audio_array = batch["path"]["array"]
        # Đảm bảo audio_array là mảng NumPy
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        # Chuẩn hóa sampling rate và padding
        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=160000
        )
        batch["input_values"] = inputs.input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcript"]).input_ids
        return batch

    dataset = dataset.map(preprocess_function, remove_columns=["path"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Tải mô hình để fine-tune
    print("Tải mô hình để fine-tune speech-to-text...")
    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        print("Mô hình đã được tải thành công!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Cấu hình huấn luyện
    training_args = TrainingArguments(
        output_dir="./wav2vec2_vietnamese_speech_to_text",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True
    )

    # Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor
    )

    # Huấn luyện và lưu mô hình
    print("Bắt đầu huấn luyện mô hình speech-to-text...")
    trainer.train()
    trainer.evaluate()
    trainer.save_model("wav2vec2_vietnamese_speech_to_text_model")
    processor.save_pretrained("wav2vec2_vietnamese_speech_to_text_model")
    print("Huấn luyện speech-to-text hoàn tất!")

# Hàm chính để chạy huấn luyện
def main():
    print("Bắt đầu huấn luyện...")
    train_speaker_identification()
    train_speech_to_text()
    print("Hoàn tất huấn luyện!")

if __name__ == "__main__":
    main()