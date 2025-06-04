import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def compare_all_experiments():
    results_dir = 'results/'
    
    if not os.path.exists(results_dir):
        print("Folder results/ nie istnieje!")
        return
    
    experiment_results = []
    
    # Przechodzimy przez wszystkie foldery w results/
    for root, dirs, files in os.walk(results_dir):
        if 'predictions.csv' in files:
            predictions_path = os.path.join(root, 'predictions.csv')
            folder_name = os.path.basename(root)
            
            print(f'\n=== Przetwarzam: {folder_name} ===')
            
            try:
                with open(predictions_path, 'r') as f:
                    lines = f.read().strip().split('\n')
                
                # Słownik do przechowywania wyników dla każdej klasy
                class_stats = {}
                
                # Przetwarzamy każdą linię (pomijamy nagłówek)
                for line in lines[1:]:
                    fields = line.split(',')
                    image_name = fields[0]
                    pred_label = int(fields[1])
                    
                    # Wyciągamy prawdziwą etykietę z nazwy pliku
                    true_label = int(image_name.split('_')[0])
                    
                    # Inicjalizujemy statystyki dla klasy jeśli nie istnieją
                    if true_label not in class_stats:
                        class_stats[true_label] = {'correct': 0, 'total': 0}
                    
                    class_stats[true_label]['total'] += 1
                    
                    # Sprawdzamy czy predykcja jest poprawna
                    if pred_label == true_label:
                        class_stats[true_label]['correct'] += 1
                
                # Obliczamy ogólną dokładność
                total_correct = sum(stats['correct'] for stats in class_stats.values())
                total_samples = sum(stats['total'] for stats in class_stats.values())
                overall_accuracy = (total_correct / total_samples) * 100
                
                # Wyświetlamy szczegółowe informacje dla każdej klasy
                print(f'Szczegółowe wyniki dla {folder_name}:')
                print('-' * 60)
                for cls in sorted(class_stats.keys()):
                    correct = class_stats[cls]['correct']
                    total = class_stats[cls]['total']
                    accuracy = (correct / total) * 100
                    print(f'  Klasa {cls}: {correct:3d}/{total:3d} = {accuracy:6.2f}%')
                
                print('-' * 60)
                print(f'  OGÓŁEM: {total_correct:3d}/{total_samples:3d} = {overall_accuracy:6.2f}%')
                
                # Obliczamy dokładność dla każdej klasy
                class_accuracies = {}
                for cls in sorted(class_stats.keys()):
                    accuracy = (class_stats[cls]['correct'] / class_stats[cls]['total']) * 100
                    class_accuracies[cls] = accuracy
                
                experiment_results.append({
                    'name': folder_name,
                    'overall_accuracy': overall_accuracy,
                    'class_accuracies': class_accuracies,
                    'total_samples': total_samples,
                    'total_correct': total_correct
                })
                
            except Exception as e:
                print(f'Błąd podczas przetwarzania {predictions_path}: {e}')
    
    if not experiment_results:
        print("Nie znaleziono żadnych plików predictions.csv!")
        return
    
    # Sortujemy eksperymenty według nazwy (chronologicznie)
    experiment_results.sort(key=lambda x: x['name'])
    
    print(f'\n\n=== PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW ===')
    print('=' * 80)
    for i, exp in enumerate(experiment_results, 1):
        print(f'{i:2d}. {exp["name"]:35s} - {exp["total_correct"]:3d}/{exp["total_samples"]:3d} = {exp["overall_accuracy"]:6.2f}%')
    
    # Tworzymy wykres liniowy porównujący ogólną dokładność
    plt.figure(figsize=(18, 10))  # Większy wykres
    
    names = [exp['name'] for exp in experiment_results]
    accuracies = [exp['overall_accuracy'] for exp in experiment_results]
    
    # Wykres liniowy
    plt.plot(range(len(names)), accuracies, 'o-', linewidth=2, markersize=8, 
             color='blue', markerfacecolor='lightblue', markeredgecolor='navy')
    
    # Ustawienia dla osi X - pokazujemy co n-tą etykietę
    step = max(1, len(names) // 10)  # Maksymalnie 10 etykiet
    indices_to_show = range(0, len(names), step)
    if len(names) - 1 not in indices_to_show:  # Zawsze pokazuj ostatnią
        indices_to_show = list(indices_to_show) + [len(names) - 1]
    
    # Dodajemy wartości tylko przy wybranych punktach (żeby się nie nakładały)
    for i in indices_to_show:
        plt.annotate(f'{accuracies[i]:.1f}%', (i, accuracies[i]), textcoords="offset points", 
                    xytext=(0,15), ha='center', fontweight='bold', fontsize=10)
    
    plt.xlabel('Eksperymenty (chronologicznie)', fontsize=12)
    plt.ylabel('Ogólna dokładność (%)', fontsize=12)
    plt.title('Porównanie ogólnej dokładności wszystkich eksperymentów', fontsize=14, pad=20)
    
    plt.xticks(indices_to_show, [names[i] for i in indices_to_show], 
               rotation=45, ha='right', fontsize=10)
    plt.subplots_adjust(bottom=0.25)  # Jeszcze więcej miejsca na dole
    
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # Dodajemy linię z średnią dokładnością
    avg_accuracy = np.mean(accuracies)
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7, 
                label=f'Średnia: {avg_accuracy:.2f}%')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/overall_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Zapisujemy szczegółowe statystyki do pliku
    with open('results/experiment_summary.txt', 'w') as f:
        f.write('PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW\n')
        f.write('=' * 50 + '\n\n')
        
        f.write('CHRONOLOGICZNA LISTA EKSPERYMENTÓW:\n')
        f.write('-' * 50 + '\n')
        for i, exp in enumerate(experiment_results, 1):
            f.write(f'{i:2d}. {exp["name"]:30s} - {exp["total_correct"]:3d}/{exp["total_samples"]:3d} = {exp["overall_accuracy"]:6.2f}%\n')
        
        # Ranking według dokładności
        sorted_results = sorted(experiment_results, key=lambda x: x['overall_accuracy'], reverse=True)
        f.write('\n\nRANKING WEDŁUG DOKŁADNOŚCI:\n')
        f.write('-' * 50 + '\n')
        for i, exp in enumerate(sorted_results, 1):
            f.write(f'{i:2d}. {exp["name"]:30s} - {exp["overall_accuracy"]:6.2f}%\n')
        
        f.write('\n\nSZCZEGÓŁOWE WYNIKI:\n')
        f.write('-' * 50 + '\n')
        for exp in experiment_results:
            f.write(f'\n{exp["name"]}:\n')
            f.write(f'  Ogólna dokładność: {exp["total_correct"]:3d}/{exp["total_samples"]:3d} = {exp["overall_accuracy"]:6.2f}%\n')
            f.write('  Dokładność dla każdej klasy:\n')
            for cls in sorted(exp['class_accuracies'].keys()):
                f.write(f'    Klasa {cls}: {exp["class_accuracies"][cls]:6.2f}%\n')
        
        # Statystyki ogólne
        best_exp = max(experiment_results, key=lambda x: x['overall_accuracy'])
        worst_exp = min(experiment_results, key=lambda x: x['overall_accuracy'])
        avg_acc = np.mean([exp['overall_accuracy'] for exp in experiment_results])
        
        f.write(f'\n\nSTATYSTYKI OGÓLNE:\n')
        f.write('-' * 50 + '\n')
        f.write(f'Liczba eksperymentów: {len(experiment_results)}\n')
        f.write(f'Najlepszy wynik: {best_exp["name"]} ({best_exp["overall_accuracy"]:.2f}%)\n')
        f.write(f'Najgorszy wynik: {worst_exp["name"]} ({worst_exp["overall_accuracy"]:.2f}%)\n')
        f.write(f'Średnia dokładność: {avg_acc:.2f}%\n')
    
    print(f'\n=== WYGENEROWANE PLIKI ===')
    print(f'  - results/overall_accuracy_comparison.png (wykres liniowy ogólnej dokładności)')
    print(f'  - results/experiment_summary.txt (szczegółowe statystyki)')
    
    best_exp = max(experiment_results, key=lambda x: x['overall_accuracy'])
    worst_exp = min(experiment_results, key=lambda x: x['overall_accuracy'])
    avg_acc = np.mean([exp['overall_accuracy'] for exp in experiment_results])
    
    print(f'\n=== STATYSTYKI KOŃCOWE ===')
    print(f'Liczba eksperymentów: {len(experiment_results)}')
    print(f'Najlepszy eksperyment: {best_exp["name"]} ({best_exp["overall_accuracy"]:.2f}%)')
    print(f'Najgorszy eksperyment: {worst_exp["name"]} ({worst_exp["overall_accuracy"]:.2f}%)')
    print(f'Średnia dokładność: {avg_acc:.2f}%')

if __name__ == "__main__":
    compare_all_experiments()