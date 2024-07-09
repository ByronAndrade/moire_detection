import numpy as np
import pywt

def fwdHaarDWT2D_pywt(img):
    # Assegurando que a imagem é um np.array e convertendo para float
    img = np.array(img, dtype=float)
    
    # Aplicando a transformada de Haar 2D
    coeffs = pywt.dwt2(img, 'haar')
    
    # Desempacotando os coeficientes
    cA, (cH, cV, cD) = coeffs
    
    # Retornando na mesma ordem que o código original: LL, LH, HL, HH
    return cA, cH, cV, cD

# Exemplo de uso:
# imagem = np.random.rand(256, 256)  # uma imagem aleatória
# LL, LH, HL, HH = fwdHaarDWT2D_pywt(imagem)
# Agora LL, LH, HL, e HH são acessíveis como no método anterior
