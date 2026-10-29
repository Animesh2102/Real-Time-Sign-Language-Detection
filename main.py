import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import customtkinter as ctk
import os
import threading
import time
from PIL import Image, ImageTk

# ------------------ PATHS & SETTINGS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ------------------ LOAD MODEL ------------------
def get_model_path(language):
    return os.path.join(BASE_DIR, f"{language}_best_model.p")

model_dict = pickle.load(open(get_model_path('ISL'), 'rb'))
model = model_dict['model']

# ------------------ MEDIAPIPE ------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

labels_dict = {
    'ISL': {i: chr(65 + i) for i in range(26)} | {26: ' ', 27: '.'},
    'ASL': {i: chr(65 + i) for i in range(26)} | {26: ' ', 27: '.'}
}

expected_features = 84
stabilization_buffer = []
word_buffer = ""
sentence = ""
last_registered_time = time.time()
registration_delay = 1.2

# ------------------ TTS ------------------
def speak_text(text):
    if not text.strip():
        return
    def tts_thread():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=tts_thread, daemon=True).start()

def load_language_model(language):
    global model
    path = get_model_path(language)
    if os.path.exists(path):
        model_dict = pickle.load(open(path, 'rb'))
        model = model_dict['model']
        current_alphabet.set("-")
        current_confidence.set("Confidence: --%")
    else:
        print(f"Model file not found: {path}")

# ------------------ APP WINDOW ------------------
app = ctk.CTk()
app.geometry("1350x820")
app.title("Real-Time Sign Language Translator (ISL & ASL)")
app.configure(fg_color="#0e0b16")

# ------------------ VARIABLES ------------------
current_language = ctk.StringVar(value="ISL")
current_alphabet = ctk.StringVar(value="-")
current_confidence = ctk.StringVar(value="Confidence: --%")
current_word = ctk.StringVar(value="")
current_sentence = ctk.StringVar(value="")
current_camera_name = ctk.StringVar(value="Default Camera")

is_paused = False

# ------------------ GLASS CARD ------------------
def glass_card(parent, width, height):
    frame = ctk.CTkFrame(
        parent,
        width=width,
        height=height,
        corner_radius=24,
        fg_color="#181424",
        border_width=1,
        border_color="#2f2542"
    )
    frame.pack_propagate(False)
    return frame

# ------------------ MAIN LAYOUT ------------------
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(fill="both", expand=True, padx=35, pady=25)

# Left camera card
left_card = glass_card(main_frame, 660, 680)
left_card.pack(side="left", padx=20, fill="both", expand=True)

# Right dashboard card
right_card = glass_card(main_frame, 560, 680)
right_card.pack(side="right", padx=20, fill="both", expand=True)

# ------------------ VIDEO FEED ------------------
video_title = ctk.CTkLabel(
    left_card,
    text="LIVE CAMERA FEED",
    font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
    text_color="#a09cb0"
)
video_title.pack(pady=(18, 8))

video_label = ctk.CTkLabel(left_card, text="", corner_radius=16)
video_label.pack(expand=True, padx=20, pady=10)

# ------------------ RIGHT CONTENT ------------------
title = ctk.CTkLabel(
    right_card,
    text="Sign Language Translator",
    font=ctk.CTkFont(size=28, weight="bold"),
    text_color="#e0aaff"
)
title.pack(pady=(22, 12))

# Current Letter & Confidence
letter_box = ctk.CTkFrame(right_card, fg_color="#221b33", corner_radius=18, border_width=1, border_color="#3d2c5c")
letter_box.pack(fill="x", padx=30, pady=8)

ctk.CTkLabel(letter_box, text="CURRENT DETECTED SIGN",
             font=ctk.CTkFont(size=12, weight="bold"),
             text_color="#c77dff").pack(pady=(10, 0))

ctk.CTkLabel(
    letter_box,
    textvariable=current_alphabet,
    font=ctk.CTkFont(size=56, weight="bold"),
    text_color="#ff66c4"
).pack(pady=2)

ctk.CTkLabel(
    letter_box,
    textvariable=current_confidence,
    font=ctk.CTkFont(size=13),
    text_color="#9d4edd"
).pack(pady=(0, 10))

# Word & Sentence Displays
word_box = ctk.CTkFrame(right_card, fg_color="#1d172b", corner_radius=14)
word_box.pack(fill="x", padx=30, pady=6)

ctk.CTkLabel(word_box, text="Current Word:", font=ctk.CTkFont(size=13, weight="bold"), text_color="#a09cb0").pack(anchor="w", padx=14, pady=(8, 0))
ctk.CTkLabel(
    word_box,
    textvariable=current_word,
    font=ctk.CTkFont(size=22, weight="bold"),
    text_color="#ffffff",
    wraplength=480
).pack(anchor="w", padx=14, pady=(2, 8))

sentence_box = ctk.CTkFrame(right_card, fg_color="#1d172b", corner_radius=14)
sentence_box.pack(fill="x", padx=30, pady=6)

ctk.CTkLabel(sentence_box, text="Constructed Sentence:", font=ctk.CTkFont(size=13, weight="bold"), text_color="#a09cb0").pack(anchor="w", padx=14, pady=(8, 0))
ctk.CTkLabel(
    sentence_box,
    textvariable=current_sentence,
    font=ctk.CTkFont(size=18),
    text_color="#e0aaff",
    wraplength=480
).pack(anchor="w", padx=14, pady=(2, 10))

# ------------------ CAMERA HANDLING ------------------
def get_available_cameras():
    cameras = {}
    for i in range(3):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            cameras[f"Camera {i}"] = i
            cap_test.release()
    if not cameras:
        cameras["Default Camera"] = 0
    return cameras

camera_mapping = get_available_cameras()

def switch_camera(choice):
    global cap
    if cap:
        cap.release()
    index = camera_mapping.get(choice, 0)
    cap = cv2.VideoCapture(index)
    current_camera_name.set(choice)

cap = cv2.VideoCapture(camera_mapping[list(camera_mapping.keys())[0]])

# ------------------ CONTROLS ------------------
controls = ctk.CTkFrame(right_card, fg_color="transparent")
controls.pack(pady=14)

language_menu = ctk.CTkOptionMenu(
    controls,
    values=["ISL", "ASL"],
    variable=current_language,
    command=load_language_model,
    fg_color="#7b2cbf",
    button_color="#9d4edd",
    font=ctk.CTkFont(size=13, weight="bold")
)
language_menu.grid(row=0, column=0, padx=8, pady=6)

camera_menu = ctk.CTkOptionMenu(
    controls,
    values=list(camera_mapping.keys()),
    command=switch_camera,
    fg_color="#c77dff",
    button_color="#e0aaff",
    text_color="#181424",
    font=ctk.CTkFont(size=13, weight="bold")
)
camera_menu.set(list(camera_mapping.keys())[0])
camera_menu.grid(row=0, column=1, padx=8, pady=6)

def backspace_letter():
    global word_buffer
    if word_buffer:
        word_buffer = word_buffer[:-1]
        current_word.set(word_buffer)

def reset_sentence():
    global word_buffer, sentence, stabilization_buffer
    word_buffer = ""
    sentence = ""
    stabilization_buffer.clear()
    current_word.set("")
    current_sentence.set("")
    current_alphabet.set("-")
    current_confidence.set("Confidence: --%")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    pause_btn.configure(text="Resume" if is_paused else "Pause")

backspace_btn = ctk.CTkButton(
    controls, text="⌫ Delete",
    command=backspace_letter,
    width=95,
    fg_color="#3d2c5c",
    hover_color="#563d80"
)
backspace_btn.grid(row=1, column=0, padx=6, pady=8)

reset_btn = ctk.CTkButton(
    controls, text="Reset",
    command=reset_sentence,
    width=95,
    fg_color="#5a189a",
    hover_color="#7b2cbf"
)
reset_btn.grid(row=1, column=1, padx=6, pady=8)

pause_btn = ctk.CTkButton(
    controls, text="Pause",
    command=toggle_pause,
    width=95,
    fg_color="#3c096c",
    hover_color="#5a189a"
)
pause_btn.grid(row=1, column=2, padx=6, pady=8)

speak_btn = ctk.CTkButton(
    controls, text="🔊 Speak",
    command=lambda: speak_text(current_sentence.get() or current_word.get()),
    width=110,
    fg_color="#ff4da6",
    hover_color="#ff85d8",
    font=ctk.CTkFont(size=13, weight="bold")
)
speak_btn.grid(row=0, column=2, padx=8, pady=6)

# ------------------ FRAME PROCESSING ------------------
def process_frame():
    global stabilization_buffer, word_buffer, sentence, last_registered_time

    ret, frame = cap.read()
    if not ret:
        app.after(15, process_frame)
        return

    # Mirror video for natural user interaction
    frame = cv2.flip(frame, 1)

    if not is_paused:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Visual skeletal tracking overlay on frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                data_aux = []
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]

                for lm in hand_landmarks.landmark:
                    data_aux.append((lm.x - min(x_)) / max(1e-5, max(x_) - min(x_)))
                    data_aux.append((lm.y - min(y_)) / max(1e-5, max(y_) - min(y_)))

                if len(data_aux) < expected_features:
                    data_aux.extend([0] * (expected_features - len(data_aux)))
                elif len(data_aux) > expected_features:
                    data_aux = data_aux[:expected_features]

                try:
                    prediction = model.predict_proba([np.asarray(data_aux)])
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]

                    if confidence > 0.45:
                        current_confidence.set(f"Confidence: {confidence * 100:.1f}%")
                        predicted_character = labels_dict[current_language.get()][predicted_class]
                        stabilization_buffer.append(predicted_character)

                        if len(stabilization_buffer) > 25:
                            stabilization_buffer.pop(0)

                        if stabilization_buffer.count(predicted_character) > 12:
                            current_time = time.time()
                            if current_time - last_registered_time > registration_delay:
                                last_registered_time = current_time
                                current_alphabet.set(predicted_character)

                                if predicted_character == ' ':
                                    if word_buffer.strip():
                                        speak_text(word_buffer)
                                        sentence += word_buffer + " "
                                        current_sentence.set(sentence.strip())
                                    word_buffer = ""
                                    current_word.set("")
                                elif predicted_character == '.':
                                    if word_buffer.strip():
                                        speak_text(word_buffer)
                                        sentence += word_buffer + "."
                                        current_sentence.set(sentence.strip())
                                    word_buffer = ""
                                    current_word.set("")
                                else:
                                    word_buffer += predicted_character
                                    current_word.set(word_buffer)
                except Exception as e:
                    pass

    # Convert to PIL Image for Tkinter rendering
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((580, 440))
    img_tk = ImageTk.PhotoImage(image=img)

    video_label.configure(image=img_tk)
    video_label.image = img_tk

    app.after(15, process_frame)

def on_closing():
    global cap
    if cap:
        cap.release()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", on_closing)
process_frame()
app.mainloop()
