import sys
import os
import numpy as np
import concurrent.futures
import threading
from mCNN import createModel
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from PIL import Image



# Configurações do modelo e das imagens
config = {
    "width": 500,
    "height": 375,
    "depth": 1,
    "num_classes": 2,
    "batch_size": 32,  
}



def load_trained_model(weights_path, config):
    model = createModel(config["height"], config["width"], config["depth"], config["num_classes"]) 
    model.load_weights(weights_path)
    return model

def process_directory(directory, label, model):
    print(f"Iniciando processamento de {directory} na thread {threading.current_thread().name}")
    predictions = []
    component_files = {}
    for filename in os.listdir(directory):
        root_name = "_".join(filename.split('_')[:1])  # Isso agrupa as imagens pela parte do nome antes do primeiro '_'
        if root_name not in component_files:
            component_files[root_name] = []
        component_files[root_name].append(os.path.join(directory, filename))

    for root_name, components in component_files.items():
        # Separe as variantes da imagem original
        original_variants = [f for f in components if 'FLIP' not in f and '180' not in f]
        variants_180 = [f for f in components if '180' in f and 'FLIP' not in f]
        flip180_variants = [f for f in components if 'FLIP' in f]

        # Classifique cada variante
        orig_pred = classify_variant(original_variants, model)
        pred_180 = classify_variant(variants_180, model)
        flip180_pred = classify_variant(flip180_variants, model)

        # Agregue os resultados das três classificações
        if orig_pred == 1 and flip180_pred == 1 and pred_180 == 1:
            final_pred = 1  # Não tem padrão de moiré
        else:
            final_pred = 0  # Tem padrão de moiré

        predictions.append((final_pred, label))
    print(f"Finalizando processamento de {directory} na thread {threading.current_thread().name}")
    return predictions

def classify_variant(components, model):
    if not components:
        return 0  # Se a lista de componentes estiver vazia, retorna 0 por padrão
    components_sorted = sorted(components, key=lambda x: x.split('_')[-1].split('.')[0])
    return process_and_classify_image(components_sorted, model)




def process_directories_concurrent(positive_dir, negative_dir, model):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: process_directory(x[0], x[1], model), [(positive_dir, 1), (negative_dir, 0)]))
    labels, predictions = zip(*[item for sublist in results for item in sublist])
    return labels, predictions



def prepare_image(path):
    img = Image.open(path)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Adiciona a dimensão do canal
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona a dimensão do batch
    return img_array



def process_and_classify_image(component_paths, model):
    components = [None] * 4
    order = {'LL': 0, 'LH': 1, 'HL': 2, 'HH': 3}

    for path in component_paths:
        suffix = path.split('_')[-1][:-5]  # Assume que a extensão tem cinco caracteres (".tiff")
        # suffix = os.path.splitext(path)[0].split('_')[-1] #opção para qualque extensão
        index = order[suffix]
        components[index] = prepare_image(path)  # Chama a nova função para preparar a imagem

    if any(x is None for x in components):
        print("Erro: Algum componente não foi carregado corretamente.")
        return None


    prediction = model.predict([components[0], components[1], components[2], components[3]])
   #for i, comp in enumerate(components):
   #    print(f"Componente {['LL', 'LH', 'HL', 'HH'][3 - i]} ({component_paths[i].split('_')[-1][:-5]}) enviado com shape: {comp.shape}")

    return 1 if prediction[0][0] > 0.5 else 0




def print_and_save_confusion_matrix(cm, true_labels, predicted_labels, file_name, model_path):
    # Detalhes para imprimir a matriz de confusão de forma mais clara
    print("Matriz de Confusão:")
    print(f"Verdadeiro Positivo (VP): {cm[1][1]}")
    print(f"Falso Negativo (FN): {cm[1][0]}")
    print(f"Falso Positivo (FP): {cm[0][1]}")
    print(f"Verdadeiro Negativo (VN): {cm[0][0]}")

    # Calculando métricas
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Imprimindo métricas
    print(f"Precisão: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Medida F1: {f1:.2f}")

    # Salva a matriz de confusão e o modelo usado em um arquivo de texto
    with open(file_name, 'w') as f:
        f.write(f"Modelo usado: {model_path}\n")
        f.write("Matriz de Confusão:\n")
        f.write(f"VP: {cm[1][1]} FN: {cm[1][0]}\n")
        f.write(f"FP: {cm[0][1]} VN: {cm[0][0]}\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Por favor, forneça os diretórios das imagens 'positive' e 'negative' e o caminho do modelo.")
    else:
        print("Carregando modelo...")
        model = load_trained_model(sys.argv[3], config)
        print("Modelo carregado. Processando diretórios...")

        # Extrai o nome base do arquivo do modelo
        model_name = os.path.splitext(os.path.basename(sys.argv[3]))[0]
        output_file_name = f'matriz_de_confusao_{model_name}.txt'

        true_labels, predicted_labels = process_directories_concurrent(sys.argv[1], sys.argv[2], model)
        cm = confusion_matrix(true_labels, predicted_labels, labels=[1, 0])
        print_and_save_confusion_matrix(cm, true_labels, predicted_labels, output_file_name, sys.argv[3])
        print(f"Matriz de confusão e métricas salvas em '{output_file_name}'.")
