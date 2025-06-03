import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history():
    # Sprawdź, czy plik istnieje
    if not os.path.exists('./models/model/training_history.json'):
        print("Plik z historią treningu nie istnieje!")
        return

    history = None    
    # Wczytaj historię treningu
    with open('./models/model/training_history.json', 'r') as f:
        history = json.load(f) 
    
    
    acc = history['acc']
    val_acc = history['val_acc']

    plt.figure(figsize=(8, 6))
    plt.plot(acc, label='Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Dokładność modelu')
    
    plt.tight_layout()
    plt.savefig('./models/model/training_history.png')
    plt.show()
    
    print(f"Najwyższa dokładność walidacji: {max(val_acc):.4f} w epoce {val_acc.index(max(val_acc))+1}")

if __name__ == "__main__":
    plot_training_history()