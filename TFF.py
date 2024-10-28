import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

# Définition du modèle de réseau de neurones simple
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Fonction d'entraînement local pour un client
def train_local_model(model, data_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# Fonction d'agrégation fédérée
def federated_averaging(models):
    avg_model = {}
    for key in models[0].keys():
        avg_model[key] = torch.stack([model[key] for model in models], dim=0).mean(dim=0)
    return avg_model

# Chargement des données et préparation des données des clients
def load_data(num_clients, poison_client=None, target_label=0, poison_label=1, test_size=0.2):
    data = load_digits()
    X = data.images.reshape(-1, 64)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Split des données parmi les clients
    client_data = []
    for i in range(num_clients):
        indices = np.random.choice(len(X_train), size=len(X_train) // num_clients, replace=False)
        X_client = torch.tensor(X_train[indices], dtype=torch.float32)
        y_client = torch.tensor(y_train[indices], dtype=torch.long)
        
        # Appliquer l'empoisonnement si c'est le client ciblé
        if i == poison_client:
            poison_indices = y_client == target_label
            y_client[poison_indices] = poison_label
        
        dataset = TensorDataset(X_client, y_client)
        client_data.append(DataLoader(dataset, batch_size=32, shuffle=True))
    
    # Préparation des données de test
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return client_data, test_loader

# Fonction d'évaluation du modèle
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Configuration de l'app Streamlit
st.title("Simulation de Federated Learning avec Attaque par Empoisonnement")
st.write("""
Simulez un système de Federated Learning avec plusieurs clients et observez l'impact d'une attaque par empoisonnement.
""")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres d'entraînement")
num_clients = st.sidebar.slider("Nombre de Clients", 2, 10, 5)
num_rounds = st.sidebar.slider("Nombre de Rounds d'Entraînement", 1, 20, 5)
epochs = st.sidebar.slider("Epochs d'Entraînement Local", 1, 10, 2)
learning_rate = st.sidebar.slider("Taux d'Apprentissage", 0.001, 0.1, 0.01, step=0.001)

# Paramètres de l'empoisonnement
st.sidebar.header("Paramètres d'Empoisonnement")
apply_poisoning = st.sidebar.checkbox("Appliquer une Attaque par Empoisonnement")
poison_client = None
target_label = 0
poison_label = 1

if apply_poisoning:
    poison_client = st.sidebar.selectbox("Sélectionner le Client à Empoisonner", range(num_clients))
    target_label = st.sidebar.number_input("Label Cible", min_value=0, max_value=9, value=0)
    poison_label = st.sidebar.number_input("Label Empoisonné", min_value=0, max_value=9, value=1)

# Chargement des données
client_data, test_loader = load_data(
    num_clients,
    poison_client=poison_client if apply_poisoning else None,
    target_label=target_label,
    poison_label=poison_label
)

# Initialisation du modèle global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = SimpleNN().to(device)
global_model_state = global_model.state_dict()

# Listes pour stocker les métriques
accuracy_history = []
accuracy_poisoned_history = []

# Entraînement fédéré
st.header("Entraînement Fédéré")
for round_num in range(num_rounds):
    st.subheader(f"Round {round_num + 1}")
    local_models = []
    for i, data_loader in enumerate(client_data):
        local_model = SimpleNN().to(device)
        local_model.load_state_dict(global_model_state)  # Partir du modèle global
        local_weights = train_local_model(local_model, data_loader, epochs, learning_rate, device)
        local_models.append(local_weights)
        
        if apply_poisoning and i == poison_client:
            st.write(f"• Client {i + 1} a été **empoisonné** (Label {target_label} → {poison_label})")
        else:
            st.write(f"• Client {i + 1} a entraîné normalement.")
    
    # Agrégation des modèles locaux
    global_model_state = federated_averaging(local_models)
    global_model.load_state_dict(global_model_state)
    st.write(f"→ Modèle global mis à jour après le round {round_num + 1}.")
    
    # Évaluation du modèle global
    accuracy = evaluate_model(global_model, test_loader, device)
    accuracy_history.append(accuracy)
    st.write(f"**Accuracy après le round {round_num + 1} : {accuracy:.2f}**")
    
    # Optionnel : évaluer avec des données empoisonnées
    if apply_poisoning:
        # Recharger les données de test avec l'empoisonnement pour cette évaluation
        data = load_digits()
        X = data.images.reshape(-1, 64)
        y = data.target
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        if poison_client is not None:
            # Simuler l'empoisonnement sur l'ensemble de validation
            y_val_poisoned = y_val.copy()
            y_val_poisoned[y_val_poisoned == target_label] = poison_label
            X_val_poisoned = torch.tensor(X_val, dtype=torch.float32)
            y_val_poisoned = torch.tensor(y_val_poisoned, dtype=torch.long)
            poisoned_val_dataset = TensorDataset(X_val_poisoned, y_val_poisoned)
            poisoned_val_loader = DataLoader(poisoned_val_dataset, batch_size=32, shuffle=False)
            poisoned_accuracy = evaluate_model(global_model, poisoned_val_loader, device)
            accuracy_poisoned_history.append(poisoned_accuracy)
            st.write(f"**Accuracy sur les données empoisonnées : {poisoned_accuracy:.2f}**")

# Visualisation des résultats
st.header("Visualisation des Performances")

# Graphique de l'accuracy globale
fig1, ax1 = plt.subplots()
ax1.plot(range(1, num_rounds + 1), accuracy_history, marker='o', label='Accuracy Globale')
if apply_poisoning:
    ax1.plot(range(1, num_rounds + 1), accuracy_poisoned_history, marker='x', label='Accuracy avec Empoisonnement')
ax1.set_xlabel('Rounds d\'Entraînement')
ax1.set_ylabel('Accuracy')
ax1.set_title('Évolution de l\'Accuracy au Fil des Rounds')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Matrice de confusion finale
st.header("Matrice de Confusion Finale")
from sklearn.metrics import confusion_matrix

# Préparer les données de test
data = load_digits()
X = data.images.reshape(-1, 64)
y = data.target
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Évaluation finale
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.long).to(device)
global_model.eval()
with torch.no_grad():
    outputs = global_model(X_val)
    _, preds = torch.max(outputs, 1)
    
cm = confusion_matrix(y_val.cpu(), preds.cpu())
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Prédictions')
plt.ylabel('Véritables Labels')
plt.title('Matrice de Confusion du Modèle Global Final')
st.pyplot(plt)

# Comparaison avec un modèle non empoisonné
if apply_poisoning:
    st.header("Comparaison avec un Modèle Non Empoisonné")
    
    # Charger les données sans empoisonnement
    client_data_clean, test_loader_clean = load_data(
        num_clients,
        poison_client=None,
        target_label=target_label,
        poison_label=poison_label
    )
    
    # Initialiser un nouveau modèle global
    clean_global_model = SimpleNN().to(device)
    clean_global_model_state = clean_global_model.state_dict()
    
    clean_accuracy_history = []
    
    # Entraînement fédéré sans empoisonnement
    for round_num in range(num_rounds):
        local_models_clean = []
        for i, data_loader in enumerate(client_data_clean):
            local_model_clean = SimpleNN().to(device)
            local_model_clean.load_state_dict(clean_global_model_state)
            local_weights_clean = train_local_model(local_model_clean, data_loader, epochs, learning_rate, device)
            local_models_clean.append(local_weights_clean)
        
        # Agrégation des modèles locaux
        clean_global_model_state = federated_averaging(local_models_clean)
        clean_global_model.load_state_dict(clean_global_model_state)
        
        # Évaluation du modèle global propre
        clean_accuracy = evaluate_model(clean_global_model, test_loader_clean, device)
        clean_accuracy_history.append(clean_accuracy)
    
    # Graphique de comparaison
    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, num_rounds + 1), accuracy_history, marker='o', label='Avec Empoisonnement')
    ax2.plot(range(1, num_rounds + 1), clean_accuracy_history, marker='s', label='Sans Empoisonnement')
    ax2.set_xlabel('Rounds d\'Entraînement')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Comparaison de l\'Accuracy avec et sans Empoisonnement')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Analyse de l'impact
    st.write("""
    **Analyse de l'Impact de l'Empoisonnement :**
    - **Avec Empoisonnement** : L'accuracy globale peut diminuer en raison de l'influence négative du client empoisonné.
    - **Sans Empoisonnement** : L'accuracy augmente de manière plus stable, montrant l'efficacité du Federated Learning en absence d'attaques.
    """)

# Solutions pour éviter les attaques par empoisonnement
st.header("Solutions pour Éviter les Attaques par Empoisonnement")
st.markdown("""
1. **Détection de Comportements Anormaux** :
    - Utiliser des techniques de détection pour identifier les clients qui envoient des mises à jour de modèles anormales.
2. **Robustesse des Méthodes d'Agrégation** :
    - Utiliser des méthodes d'agrégation robustes comme la médiane ou le trim-mean au lieu de la moyenne simple.
3. **Validation des Modèles Locaux** :
    - Évaluer la performance des modèles locaux sur un ensemble de validation avant l'agrégation.
4. **Limitation de la Contribution des Clients** :
    - Restreindre l'impact qu'un seul client peut avoir sur le modèle global en normalisant les mises à jour.
5. **Apprentissage Différentiel et Cryptographie** :
    - Utiliser des techniques avancées pour protéger la confidentialité et l'intégrité des mises à jour des modèles.
""")

st.markdown("""
---
**Mission :**

L'objectif principal du stage est l'implémentation et l'étude d'un système de Federated Learning sur des cibles contraintes, comme des microcontrôleurs 32-bit. Cette plateforme sera utilisée pour l'étude de la sécurité de ces systèmes, par exemple contre des attaques par empoisonnement. Les objectifs sont : 

1. Réaliser un état de l’art sur les méthodologies et plateformes de Federated Learning en fonction des cibles matérielles qui seront sélectionnées pour le stage (type microcontrôleurs 32-bit avec ou sans accélérateur IA (NPU, TPU…)) en fonction de leur disponibilité

2. Simuler (Python) un système de Federated Learning à partir de cas d’usage à définir et décrire (computer vision, NLP…) et reposant sur des benchmarks publics.

3. Mettre en place et documenter (gitlab) un démonstrateur fonctionnel de cross-device Federated Learning avec plusieurs cartes sélectionnées pour le stage

4. À partir d’outils et de résultats du CEA, utiliser la plateforme pour démontrer des scénarios d’attaques (par exemple, attaques par empoisonnement)
""")
