import sys #
import os #
import tensorflow as tf #
import numpy as np #
import argparse #
from tensorflow.python.client import device_lib #
from os import listdir #
from os.path import join
from PIL import Image #
from sklearn.model_selection import train_test_split #
from mCNN import createModel #
from keras.callbacks import ModelCheckpoint #
import logging #
from tensorflow.keras.callbacks import LambdaCallback
from train_lotes import load_image, parse_arguments, image_data_generator, prepare_data_paths, weighted_binary_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau




# Definindo os pesos
# Pode ajustar esses valores conforme necessário para o seu caso
zero_weight = 1
one_weight = 1


#Configurações de Logging
logging.basicConfig(filename='fine_tuning_log.txt', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s',filemode='w')


# Callback para logar o final de cada época
epoch_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: logging.info(f'Época {epoch+1} concluída. Loss: {logs["loss"]}, Accuracy: {logs["accuracy"]}')
)

# Adicionando o callback à lista de callbacks
callbacks_list = [epoch_logging_callback]


# Configurações do modelo e das imagens
config = {
    "width": 500,
    "height": 375,
    "depth": 1,
    "num_classes": 2,
}



def parse_arguments():
    parser = argparse.ArgumentParser(description="Script para fine-tuning de um modelo CNN.")
    parser.add_argument('trainDataPositive', type=str, help='Caminho para os dados de treino positivos.')
    parser.add_argument('trainDataNegative', type=str, help='Caminho para os dados de treino negativos.')
    parser.add_argument('modelPath', type=str, help='Caminho para o modelo a ser fine-tuned.')
    parser.add_argument('epochs', type=int, default=10, help='Número de épocas para o treinamento.')
    parser.add_argument('batch_size', type=int, default=32, help='Tamanho do lote para o treinamento.')
    return parser.parse_args()



def main(args):
    # Extract paths and epochs from the command line arguments
    positiveImagePath = args.trainDataPositive
    negativeImagePath = args.trainDataNegative
    numEpochs = args.epochs
    batch_size = args.batch_size

    print("Preparando caminhos de dados...")
    train_files, val_files = prepare_data_paths(args.trainDataPositive, args.trainDataNegative)
    print(f"Quantidade de arquivos de treino: {len(train_files)}, Quantidade de arquivos de validação: {len(val_files)}")
    logging.info(f"Iniciando o fine tuning do modelo com {numEpochs} épocas e tamanho de lote {batch_size}.")
    
    train_generator = image_data_generator(train_files, batch_size=batch_size)
    validation_generator = image_data_generator(val_files, batch_size=batch_size)
    
    steps_per_epoch = int(np.ceil(len(train_files) / batch_size))
    validation_steps = int(np.ceil(len(val_files) / batch_size))

    model = trainCNNModel(train_generator, steps_per_epoch, validation_generator, validation_steps, config["height"], config["width"], config["depth"], config["num_classes"], args.epochs, args.modelPath)



# Configuração da Estratégia de Múltiplas GPUs
def trainCNNModel(generator, steps_per_epoch, validation_data, validation_steps, height, width, depth, num_classes, num_epochs, model_path):

    # Set up the strategy for multi-GPU training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():

        model = load_model(model_path, custom_objects={'loss': weighted_binary_crossentropy(zero_weight, one_weight)})

        # model = load_model(model_path)

        # Para cada entrada inpLL, inpLH, inpHL, inpHH, você tem:
        # -Convolution2D
        # -BatchNormalization
        # -MaxPooling2D
        # Se seu objetivo é congelar o bloco inicial de camadas para cada uma das quatro entradas,
        # então você precisa congelar essas três camadas para cada entrada. Considerando que você tem 4 entradas, e cada entrada passa por 3 camadas como mencionado, você terá:
        # 4 entradas * 3 camadas por entrada = 12 camadas
        # Isso congelará a primeira camada de convolução, batch normalization e max pooling para cada uma das quatro entradas
        # Congelar as primeiras 'n' camadas. Você pode ajustar 'n' conforme necessário
        n = 12
        for layer in model.layers[:n]:
            layer.trainable = False

        # Verifique quais camadas estão congeladas (opcional, para verificação)
        for layer in model.layers:
            print(layer.name, layer.trainable)

        # mude os pesos para punir o modelo.  
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) #no treino foi de 0.001  
        model.compile(
            loss=weighted_binary_crossentropy(zero_weight, one_weight),
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        # Create a directory for model checkpoints
        checkPointFolder = 'checkPoint_fine_tuning'
        if not os.path.exists(checkPointFolder):
            os.makedirs(checkPointFolder)
        checkpoint_name = checkPointFolder + '/Weights-{epoch:03d}--{val_loss:.5f}.keras'
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        # Define a callback para reduzir a taxa de aprendizado quando necessário
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)


        # Integrando todos os callbacks em uma única lista
        callbacks_list = [checkpoint, epoch_logging_callback, reduce_lr]


        ########################################

        # Training the model
        print("Iniciando o treinamento do modelo...")



        model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=validation_data, validation_steps=validation_steps, callbacks=callbacks_list)


        print("Treinamento concluído.")



        ########################################


        
    # Evaluate the model on the validation data
    print("Iniciando avaliação do modelo...")
    score, acc = model.evaluate(validation_data, steps=validation_steps, verbose=1)
    print(f"Validation score: {score}, Accuracy: {acc}")
    logging.info("Treinamento concluído. Avaliando o modelo...")
    logging.info(f"Resultado da Avaliação - Score: {score}, Accuracy: {acc}")

    # Save the final model
    model.save('moirePattern_CNN_fine_tuning.keras')
    
    return model



if __name__ == '__main__':

    try:
        # Check and display the number of available GPUs.
        num_gpus = len(tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", num_gpus)

        # Optionally, list all devices detected by TensorFlow. This includes CPUs and GPUs.
        local_devices = device_lib.list_local_devices()
        print("Local devices detected:\n", local_devices)

        # Parsing command line arguments required for the main function.
        parsed_args = parse_arguments()
        
        # Execute the main function with the parsed arguments.
        main(parsed_args)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
