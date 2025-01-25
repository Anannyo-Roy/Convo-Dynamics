#WHISPER X

import whisperx
import moviepy.editor as mp

def process_audio(audio_file, device="cuda", batch_size=4, compute_type="float16"):
    audio = whisperx.load_audio(audio_file)

    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=batch_size)

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token="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",device=device)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # print(result['segments'])

    end_timings, speaker_0_sentences = get_end_timings_and_sentences_by_speaker_0(result["segments"])

    return end_timings, speaker_0_sentences

def get_end_timings_and_sentences_by_speaker_0(result_segments):
    end_timings = []
    speaker_0_sentences = []
    current_sentence = ""
    sentence_end = 0.0
    for segment in result_segments:
        if segment['speaker'].startswith('SPEAKER_01'):
            sentence_end = segment['end']
            current_sentence += " " + segment['text']
        elif sentence_end != 0.0:
            end_timings.append(sentence_end)
            speaker_0_sentences.append(current_sentence.strip())
            current_sentence = ""
            sentence_end = 0.0
    if sentence_end != 0.0:
        end_timings.append(sentence_end)
        speaker_0_sentences.append(current_sentence.strip())
    return end_timings, speaker_0_sentences

def get_audio_length(audio_file):
    audio_clip = mp.AudioFileClip(audio_file)
    duration = audio_clip.duration
    return duration

# # Example usage:
# audio_file = r"C:\Users\yashd\Downloads\Vidinsta_Instagram Post_66169582d2a28.mp4"
# end_timings, speaker_0_sentences = process_audio(audio_file)
# print("End timings of sentences said by Speaker 0:", end_timings)
# print("Sentences said by Speaker 0:", speaker_0_sentences)

# audio_length = get_audio_length(audio_file)
# print("Audio length:", audio_length, "seconds")

# print(len(end_timings), len(speaker_0_sentences))

#ROBERTA 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

def sentimentAnalysis_batch(text_list):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    encoded_texts = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**encoded_texts)
    logits = outputs.logits
    softmax_scores = softmax(logits.detach().numpy(), axis=1)

    results = []
    for score in softmax_scores:
        max_index = int(torch.argmax(torch.tensor(score)))
        sentiment_label = None
        if max_index == 0:
            sentiment_label = "Negative"
        elif max_index == 1:
            sentiment_label = "Neutral"
        elif max_index == 2:
            sentiment_label = "Positive"

        results.append(sentiment_label)

    return results

#print(sentimentAnalysis_batch(speaker_0_sentences))


#EMOTION RECOGNITION

import cv2
import numpy as np
from keras.models import model_from_json
import moviepy.editor as mp

def get_avg_emotions_for_timings(video_path, timings):
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    json_file = open("model/emotion_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    emotion_model.load_weights("model/emotion_model.h5")
    print("Loaded model from disk")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    interval_frames = int(fps * 1)  # Number of frames in 1 seconds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    avg_emotions = []

    for timing in timings:
        timing_frames = int(timing * fps)

        # Move to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, timing_frames)

        emotions_in_interval = []

        # Read frames until the end of the interval or end of video
        for _ in range(interval_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in num_faces:
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                emotions_in_interval.append(emotion_dict[maxindex])

        # Calculate average emotion for the interval
        if len(emotions_in_interval) > 0:
            avg_emotion = max(set(emotions_in_interval), key=emotions_in_interval.count)
        else:
            avg_emotion = "Neutral"  # If no face detected, consider as Neutral emotion
        avg_emotions.append(avg_emotion)

    cap.release()
    cv2.destroyAllWindows()

    return avg_emotions

# # Example usage:
# video_path = r"C:\Users\yashd\Downloads\Vidinsta_Instagram Post_66169582d2a28.mp4"
# emotions_list = get_avg_emotions_for_timings(video_path, end_timings)
# print("Average emotions for provided timings:", emotions_list)

# print(len(end_timings), len(emotions_list))

#MAIN MAIN MAIN

from collections import Counter

def calculate_average_emotion(emotion_list):
    emotion_counts = Counter(emotion_list)
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    return most_common_emotion

def analyze_video(video_path, audio_path=None):
    if audio_path:
        # If audio path is provided, process audio from the audio file
        end_timings, speaker_0_sentences = process_audio(audio_path)
    else:
        # Otherwise, process audio directly from the video file
        end_timings, speaker_0_sentences = process_audio(video_path)

    # Step 2: Get predicted emotions for the timings (Assuming sentimentAnalysis_batch function is defined elsewhere)
    predicted_emotions = sentimentAnalysis_batch(speaker_0_sentences)

    # Step 3: Get actual emotions shown (Assuming get_avg_emotions_for_timings function is defined elsewhere)
    actual_emotions = get_avg_emotions_for_timings(video_path, end_timings)

    # Combine the results into a 2D list
    result_list = []
    for i in range(len(end_timings)):
        result_list.append([speaker_0_sentences[i], predicted_emotions[i], actual_emotions[i]])

    average_predicted_emotion_shown = calculate_average_emotion(predicted_emotions)
    average_actual_emotion_shown = calculate_average_emotion(actual_emotions)

    return result_list, [average_predicted_emotion_shown,average_actual_emotion_shown]

# Example usage:
