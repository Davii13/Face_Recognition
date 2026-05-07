import cv2
import face_recognition
import os
import numpy as np
import threading


class WebcamStream:

    def __init__(self, src=0):

        self.stream = cv2.VideoCapture(src)

        # RESOLUÇÃO DA EXIBIÇÃO
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # REDUZ ATRASO
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()

        self.stopped = False

    def start(self):

        threading.Thread(
            target=self.update,
            daemon=True
        ).start()

        return self

    def update(self):

        while not self.stopped:

            self.grabbed, self.frame = self.stream.read()

    def read(self):

        return self.frame

    def stop(self):

        self.stopped = True
        self.stream.release()


def carregar_rostos():

    diretorio = os.path.dirname(os.path.abspath(__file__))

    pasta = os.path.join(diretorio, "rostos_conhecidos")

    if not os.path.exists(pasta):

        os.makedirs(pasta)

    encodings_conhecidos = []
    nomes_conhecidos = []

    for arquivo in os.listdir(pasta):

        if arquivo.endswith((".jpg", ".jpeg", ".png")):

            caminho = os.path.join(pasta, arquivo)

            nome = os.path.splitext(arquivo)[0]

            try:

                imagem = face_recognition.load_image_file(caminho)

                encoding = face_recognition.face_encodings(imagem)

                if len(encoding) > 0:

                    encodings_conhecidos.append(encoding[0])

                    nomes_conhecidos.append(nome)

                    print(f"{nome} carregado!")

            except:

                print(f"Erro ao carregar {arquivo}")

    return encodings_conhecidos, nomes_conhecidos


def reconhecer(
    frame,
    encodings_conhecidos,
    nomes_conhecidos
):

    # FRAME PEQUENO APENAS PARA IA
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )

    rgb_small = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )

    locais = face_recognition.face_locations(
        rgb_small,
        model="hog"
    )

    encodings = face_recognition.face_encodings(
        rgb_small,
        locais
    )

    nomes = []

    for face_encoding in encodings:

        nome = "Desconhecido"

        if len(encodings_conhecidos) > 0:

            distancias = face_recognition.face_distance(
                encodings_conhecidos,
                face_encoding
            )

            melhor = np.argmin(distancias)

            if distancias[melhor] < 0.60:

                nome = nomes_conhecidos[melhor]

        nomes.append(nome)

    return locais, nomes


def desenhar(
    frame,
    locais,
    nomes
):

    for (top, right, bottom, left), nome in zip(locais, nomes):

        # VOLTA ESCALA
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.rectangle(
            frame,
            (left, bottom - 30),
            (right, bottom),
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.putText(
            frame,
            nome,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1
        )


def main():

    print("Iniciando sistema...")

    encodings_conhecidos, nomes_conhecidos = carregar_rostos()

    webcam = WebcamStream().start()

    locais = []
    nomes = []

    contador = 0

    while True:

        frame = webcam.read()

        if frame is None:
            continue

        # ESPELHO
        frame = cv2.flip(frame, 1)

        # PROCESSA APENAS A CADA 8 FRAMES
        if contador % 8 == 0:

            locais, nomes = reconhecer(
                frame,
                encodings_conhecidos,
                nomes_conhecidos
            )

        contador += 1

        desenhar(
            frame,
            locais,
            nomes
        )

        cv2.imshow(
            "Reconhecimento Facial",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    webcam.stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()