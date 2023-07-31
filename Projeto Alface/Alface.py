import cv2
import numpy as np

#Carrega o vídeo alface.mp4
video = cv2.VideoCapture('alface.mp4')
contador = 0
liberado = False
pausado = False

while True:
    if not pausado:
        ret, img = video.read()
        img = cv2.resize(img, (1100, 720)) #Redimenciona a Imagem para largura: 1100 e altura: 720
        imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #Posiciona o retângulo de contagem na parte inferior
        x, y, w, h = 30, img.shape[0] - 60, img.shape[1] - 40, 20

        #Serve para segmentar a imagem através dos pixels usando limiar adaptativo
        imgTh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
        kernel = np.ones((8, 8), np.uint8)
        imgDil = cv2.dilate(imgTh, kernel, iterations=2)

        #Recorta a região de interesse para contar os pixels brancos
        recorte = imgDil[y:y+h, x:x+w]
        brancos = cv2.countNonZero(recorte)

        #Verifica se a contagem é valida e implementa o contador, e verifica se a região está liberada para contagem.
        if brancos > 17000 and liberado == True:
            contador += 1
        if brancos < 17600:
            liberado = True
        else:
            liberado = False

        #Muda a cor do retangulo dependendo se estiver livre a passagem
        if liberado == False:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)

        #Desenha o retangulo
        cv2.rectangle(imgTh, (x, y), (x + w, y + h), (255, 255, 255), 6)
        cv2.putText(img, str(brancos), (x - 30, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        #Exibe a contagem dos alfaces na parte superior
        cv2.putText(img, str(contador), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        
        cv2.imshow('video original', img)

    #Armazena em um arquivo de texto a quantidade de alfaces que foi contado
    with open('quantidade_alfaces.txt', 'w') as arquivo:
        arquivo.write(f"Quantidade de alfaces contadas: {contador}")

    #Cria uma tecla "P" para pause e retornar, e a tecla "Q" para encerrar
    key = cv2.waitKey(20)
    if key == ord('p') or key == ord('P'):
        pausado = not pausado 
    if key == ord('q') or key == ord('Q'):
        break

video.release()
cv2.destroyAllWindows()
