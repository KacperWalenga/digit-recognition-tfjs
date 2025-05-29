import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Ścieżka do katalogu results/
results_dir = 'results/'

# Przeszukaj wszystkie podkatalogi w results/
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file == 'predictions.csv':
            predictions_path = os.path.join(root, file)
            output_path = os.path.join(root, 'confusion_matrix.png')

            # Sprawdź, czy plik już istnieje
            if os.path.exists(output_path):
                #print(f'Plik {output_path} już istnieje, pomijam generowanie.')
                continue  # pomiń ten katalog, przejdź do następnego

            print(f'Przetwarzam: {predictions_path}')
            
            # Wczytaj dane z predictions.csv
            with open(predictions_path, 'r') as f:
                lines = f.read().strip().split('\n')
            
            all_labels = []
            all_pred_labels = []

            for line in lines[1:]:  # pomijamy nagłówek
                fields = line.split(',')
                image_name = fields[0]
                pred_label = int(fields[1])
                
                true_label = int(image_name.split('_')[0])
                all_labels.append(true_label)
                all_pred_labels.append(pred_label)
            
            all_labels_np = np.array(all_labels)
            all_pred_labels_np = np.array(all_pred_labels)
            
            # Wykres
            fig, ax = plt.subplots(figsize=(8, 8))
            ConfusionMatrixDisplay.from_predictions(all_labels_np, all_pred_labels_np, ax=ax)
            plt.title('Macierz pomyłek')
            
            # Zapisz w tym samym katalogu
            plt.savefig(output_path)
            plt.close()
            print(f'Zapisano macierz: {output_path}')
