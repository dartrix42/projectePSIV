import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import shutil
import json

IMG_SIZE = 128


model = tf.keras.models.load_model("model_emocions.keras")



with open("classes.json", "r", encoding="utf-8") as f:
    clases = json.load(f)

print("Classes carregades:", clases)


carpeta_audios = "TEST_RAVDESS"


carpeta_detectadas = "detectadas"


os.makedirs(carpeta_detectadas, exist_ok=True)


for classe in clases:
    os.makedirs(os.path.join(carpeta_detectadas, classe), exist_ok=True)


extensiones_audio = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


for archivo in os.listdir(carpeta_audios):

    if not archivo.lower().endswith(extensiones_audio):
        continue

    ruta_audio = os.path.join(carpeta_audios, archivo)

    try:

        audio, sr = librosa.load(ruta_audio, sr=48000)


        audio = librosa.util.normalize(audio)


        DURACION = 3
        longitud_objetivo = sr * DURACION

        if len(audio) > longitud_objetivo:
            audio = audio[:longitud_objetivo]
        else:
            audio = np.pad(audio, (0, longitud_objetivo - len(audio)))


        S = librosa.stft(audio, n_fft=2048, hop_length=512)

        S_db = librosa.amplitude_to_db(
            np.abs(S),
            ref=np.max
        )

        plt.figure(figsize=(4, 4))
        ruta_img = "temp_test.png"
        librosa.display.specshow(
            S_db,
            sr=sr,
            cmap="magma"
        )

        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.savefig(
            ruta_img,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()


        img = tf.keras.preprocessing.image.load_img(
            ruta_img,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)


        prediccio = model.predict(img_array, verbose=0)

        index = np.argmax(prediccio)
        emocion = clases[index]
        probabilidad = prediccio[0][index]

        print("--------------------------------")
        print("Audio:", archivo)
        print("Predicció:", emocion)
        print("Probabilitat:", round(probabilidad * 100, 2), "%")


        carpeta_destino = os.path.join(carpeta_detectadas, emocion)
        ruta_destino = os.path.join(carpeta_destino, archivo)

        shutil.copy2(ruta_audio, ruta_destino)


        if os.path.exists(ruta_img):
            os.remove(ruta_img)

    except Exception as e:
        print("Error amb l'àudio:", archivo)
        print(e)