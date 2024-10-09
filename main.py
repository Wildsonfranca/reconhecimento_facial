import cv2
import numpy as np
import os
import flet as ft

def captura(largura, altura):
    #classificadores
    classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    classificador_olho = cv2.CascadeClassifier('haarcascade_eye.xml')

    #abre camera
    camera = cv2.VideoCapture(0)

    amostra = 1
    n_amostras = 25
    id = input('Digite o ID do usuário: ')
    print('Capturando as imagens...')

    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150,150))

        for (x, y, l, a) in faces_detectadas:
            cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
            regiao = imagem[y:y + a, x:x + l]
            regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
            olhos_detectados = classificador_olho.detectMultiScale(regiao_cinza_olho)

            for (ox, oy, ol, oa) in olhos_detectados:
                cv2.rectangle(regiao, (ox, oy), (ox+ol, oy+oa), (0, 255,0), 2)

            if np.average(imagem_cinza) > 110 and amostra <= n_amostras:
                imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite(f'fotos/pesso.{str(id)}.{str(amostra)}.jpg', imagem_face)
                print(f'[foto] {str(amostra)} capturada com sucesso')
                amostra += 1

        cv2.imshow('Detectar faces', imagem)    
        cv2.waitKey(1)

        if (amostra >= n_amostras + 1):
            print('Faces capturadas com sucesso.')
            break
        elif cv2.waitKey(1) == ord('q'):
            print('Camera encerrada.')
            break
            
    camera.release()
    cv2.destroyAllWindows()

def get_imagem_com_id():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagem_face)
    return np.array(ids), faces

def treinamento():
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()
    ids, faces = get_imagem_com_id()
    print('Treinando...')
    eigenface.train(faces, ids)
    eigenface.write('classificadorEigen.yml')
    fisherface.train(faces, ids)
    fisherface.write('classificadorFisher.yml')
    lbph.train(faces, ids)
    lbph.write('classificadorLBPH.yml')
    print('Treinamento finalizado com sucesso!')

def reconhecedor_aigenfaces(largura, altura):
    detector_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    reconhecedor = cv2.face.EigenFaceRecognizer_create()
    reconhecedor.read("classificadorEigen.yml")
    fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
    
    camera = cv2.VideoCapture(0)
    
    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        faces_detectadas = detector_faces.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30,30))
        
        for (x,y,l,a) in faces_detectadas:
            imagem_face = cv2.resize(imagem_cinza[y:y+a, x:x+l], (largura, altura))
            cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
            id, confianca = reconhecedor.predict(imagem_face)
            cv2.putText(imagem, str(id), (x,y+(a+30)), fonte, 2, (0,0,255))
        
        cv2.imshow("Reconhecer faces", imagem)
        if cv2.waitKey(1) == ord("q"):
            break
            
    camera.release()
    cv2.destroyAllWindows()    

# Função para a interface gráfica
def main(page: ft.Page):
    page.title = "Sistema de Reconhecimento Facial"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    # Função dos botões
    def on_captura_click(e):
        captura(220, 220)

    def on_treinamento_click(e):
        treinamento()

    def on_reconhecer_click(e):
        reconhecedor_aigenfaces(220, 220)

    # Botões
    btn_captura = ft.ElevatedButton("Capturar Imagem", on_click=on_captura_click)
    btn_treinamento = ft.ElevatedButton("Treinar Sistema", on_click=on_treinamento_click)
    btn_reconhecer = ft.ElevatedButton("Reconhecer Faces", on_click=on_reconhecer_click)

    # Adiciona os botões à página
    page.add(btn_captura, btn_treinamento, btn_reconhecer)

# Rodar a aplicação
ft.app(target=main)
