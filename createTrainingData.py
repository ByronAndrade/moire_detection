import sys
import argparse
import os
import numpy as np
from PIL import Image
from haar2D import fwdHaarDWT2D_pywt
import multiprocessing
import logging
from tqdm import tqdm
import psutil


# Configurações do modelo e das imagens
config = {
    "width": 500,
    "height": 375,
}


# Configuração de logging
logging.basicConfig(level=logging.INFO, filename='process_log.txt', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)








def process_image(args):
    f, mainFolder, trainFolder = args
    processed_count = 0
    try:
        img = Image.open(os.path.join(mainFolder, f)) 
        img = img.resize((2 * config['width'], 2 * config['height']))  # Escolha o tamanho adequado para suas imagens
        imgGray = img.convert('L')

        for transform, suffix in [(None, ''), (Image.ROTATE_180, '180_'), (Image.FLIP_LEFT_RIGHT, '180_FLIP_')]:
            transformed_img = imgGray if transform is None else imgGray.transpose(transform)
            if transform_image_and_save(transformed_img, f, suffix, trainFolder):
                processed_count += 1  # Incrementa somente se a imagem foi salva com sucesso

    except Exception as e:
        logging.error(f'Error processing {f}: {e}')
        return 0  # Retorna 0 para indicar falha no processamento dessa imagem

    return processed_count  # Retorna o número de imagens processadas com sucesso

 
def normalize_and_convert_image(component):
    component_norm = (component - component.min()) / (component.max() - component.min()) * 255
    return Image.fromarray(component_norm.astype(np.uint8))



def transform_image_and_save(image, f, customStr, path):
    try:
        cA, cH, cV, cD = fwdHaarDWT2D_pywt(image)  # Chama a função de transformação
        base_fname = os.path.splitext(f)[0]
        formats = ['LL', 'LH', 'HL', 'HH']
        components = [cA, cH, cV, cD]

        for comp, fmt in zip(components, formats):
            comp_img = normalize_and_convert_image(comp)  # Normaliza e converte cada componente
            comp_img.save(os.path.join(path, f'{base_fname}_{customStr}{fmt}.tiff'))  # Salva a imagem
        return True
    except Exception as e:
        logging.error(f'Error saving transformed image {f}_{customStr}{fmt}.tiff: {e}')
        return False




def createTrainingData(imagePath, trainFolderPath):
    try:
        imageFiles = [f for f in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, f))]
        if not os.path.exists(trainFolderPath):
            os.makedirs(trainFolderPath)
        logging.info(f'Start processing {len(imageFiles)} images in {imagePath}')

        pool = multiprocessing.Pool()
        results = []

        # Configurando a barra de progresso com descrição e comportamento após conclusão
        with tqdm(total=len(imageFiles), desc="Processing Images", leave=False, ncols=100) as pbar:
            for result in pool.imap_unordered(process_image, [(f, imagePath, trainFolderPath) for f in imageFiles]):
                results.append(result)
                pbar.update()  # Atualiza a barra de progresso
                # Monitorar a utilização da CPU a cada 10 iterações para reduzir o número de logs
                if len(results) % 10 == 0:
                    cpu_usage = psutil.cpu_percent(interval=None, percpu=True)
                    logging.info(f'CPU Usage: {cpu_usage}')

        pool.close()
        pool.join()

        logging.info(f'Finished processing images in {imagePath}')
        return sum(results)
    except Exception as e:
        logging.error(f'Error creating training data: {e}')
        return 0

def main(args):
    try:
        if args.train == 0:
            trainFolders = ('./trainDataPositive', './trainDataNegative')
        else:
            trainFolders = ('./testDataPositive', './testDataNegative')

        total_positive = createTrainingData(args.positiveImages, trainFolders[0])
        total_negative = createTrainingData(args.negativeImages, trainFolders[1])

        logging.info(f'Total positive files after augmentation: {total_positive}')
        logging.info(f'Total negative files after augmentation: {total_negative}')
    except Exception as e:
        logging.error(f'Unhandled exception in main: {e}')

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('positiveImages', type=str, help='Directory with positive (Normal) images.')
    parser.add_argument('negativeImages', type=str, help='Directory with negative (Moiré pattern) images.')
    parser.add_argument('train', type=int, help='0 = train, 1 = test')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
