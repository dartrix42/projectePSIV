import os
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import librosa.display


base_path = "Audio_Speech_Actors_01-24"

for actor in os.listdir(base_path):
    actor_path = os.path.join(base_path, actor)

    if os.path.isdir(actor_path):

        for archivo in os.listdir(actor_path):
            if archivo.endswith(".wav"):
                ruta = os.path.join(actor_path, archivo)

                audio, sr = librosa.load(ruta, sr=48000)

                print("Procesando:", ruta)

                parts = archivo.split("-")
                emocio = parts[2]

                if emocio == "01"  or emocio == "02":
                    etiqueta = "calmat"
                elif emocio == "03":
                    etiqueta = "content"
                elif emocio == "04":
                    etiqueta = "trist"
                elif emocio == "05":
                    etiqueta = "enfadat"
                elif emocio == "06":
                    etiqueta = "espantat"
                elif emocio == "07":
                    etiqueta = "fastic"
                elif emocio == "08":
                    etiqueta = "sorpresa"
                else:
                    raise ValueError

                output_dir = os.path.join("plots", etiqueta)
                os.makedirs(output_dir, exist_ok=True)

                # Normalizar volumen
                audio = librosa.util.normalize(audio)

                # Duración fija de 3 segundos
                DURACION = 3
                longitud_objetivo = sr * DURACION

                if len(audio) > longitud_objetivo:
                    audio = audio[:longitud_objetivo]
                else:
                    audio = np.pad(audio, (0, longitud_objetivo - len(audio)))

                # Crear espectrograma
                plt.figure(figsize=(4, 4))

                S = librosa.stft(audio, n_fft=2048, hop_length=512)
                S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

                # Mostrar solo el espectrograma limpio
                librosa.display.specshow(S_db, sr=sr, cmap="magma")

                plt.axis("off")
                plt.tight_layout(pad=0)


                nombre_base = os.path.splitext(archivo)[0]
                nombre_plot = f"Espectrograma_{nombre_base}.png"

                ruta_guardado = os.path.join(output_dir, nombre_plot)

                plt.savefig(
                    ruta_guardado,
                    bbox_inches="tight",
                    pad_inches=0
                )
                plt.close()