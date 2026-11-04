import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.composer.symbolic_tokenizer import SymbolicMultitrackTokenizer
from src.composer.multitrack_transformer import MultitrackCrossAttentionTransformer

def build_training_dataset():
    """
    Construye las secuencias de entrenamiento a partir de los 8 stems
    y los análisis compás a compás de las 3 canciones del dataset.
    """
    tokenizer = SymbolicMultitrackTokenizer()
    processed_files = [
        "data/processed/cumbia_001_analysis.json",
        "data/processed/cumbia_002_analysis.json",
        "data/processed/cumbia_003_analysis.json"
    ]

    all_sequences = []

    for p in processed_files:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        song_id = data["song_id"]
        key = data.get("tonalidad_predominante", "Am")
        tramos = data.get("tramos", [])
        matriz = data.get("matriz_arreglo_compases", [])

        # Convertir compases y actividades en secuencias de eventos
        events = []
        for bar_data in matriz:
            bar_idx = bar_data["bar"]
            active_inst = bar_data.get("instrumentos_activos", [])
            has_vox = bar_data.get("has_vocals", False)
            has_bombo = bar_data.get("has_bombo", False)
            has_lead = bar_data.get("has_lead", False)
            chord = bar_data.get("chord", key)

            # Determinar sección
            sec = "VERSO"
            for t in tramos:
                if t["compas_inicio"] <= bar_idx <= t["compas_fin"]:
                    sec = t["tipo"]
                    break
            
            # Eventos por instrumento
            # Güiro: activo siempre en cumbia
            for step in range(0, 16, 2):
                events.append({"track": "01_guiro", "pitch": 70, "step": step, "duration": 1, "velocity": 90, "section": sec})

            # Bombo
            if has_bombo or sec in ["Estribillo", "Tramo_Bailable_Punteo"]:
                for step in [0, 4, 8, 12]:
                    events.append({"track": "02_bombo", "pitch": 36, "step": step, "duration": 2, "velocity": 110, "section": sec})

            # Timbales
            if "timbales" in active_inst or sec == "Estribillo":
                for step in [2, 6, 10, 14]:
                    events.append({"track": "03_timbales", "pitch": 76, "step": step, "duration": 1, "velocity": 95, "section": sec})

            # Bajo
            events.append({"track": "04_bass", "pitch": 36, "step": 0, "duration": 3, "velocity": 100, "section": sec})
            events.append({"track": "04_bass", "pitch": 43, "step": 6, "duration": 3, "velocity": 95, "section": sec})
            events.append({"track": "04_bass", "pitch": 48, "step": 10, "duration": 4, "velocity": 105, "section": sec})

            # Piano / Teclados
            for step in [2, 6, 10, 14]:
                events.append({"track": "05_piano", "pitch": 60, "step": step, "duration": 2, "velocity": 85, "section": sec})

            # Lead (Acordeón o Viola)
            if has_lead:
                for step in [0, 2, 4, 6, 8, 10, 12, 14]:
                    events.append({"track": "07_lead", "pitch": 72, "step": step, "duration": 2, "velocity": 110, "section": sec})
            elif not has_vox and sec == "Verso":
                # Contestación solista en silencio de voz (Llamada y respuesta)
                for step in [2, 5, 8, 11]:
                    events.append({"track": "07_lead", "pitch": 76, "step": step, "duration": 2, "velocity": 115, "section": "CONTESTACION"})

            # Voz
            if has_vox:
                for step in [0, 4, 8, 12]:
                    events.append({"track": "08_vocals", "pitch": 67, "step": step, "duration": 3, "velocity": 95, "section": sec})

        token_matrix = tokenizer.tokenize_score(events)
        all_sequences.append(token_matrix)

    return all_sequences

def train_cumbia_transformer(epochs=60, lr=1e-3, save_path="models/cumbia_transformer_weights.pt"):
    os.makedirs("models", exist_ok=True)
    tokenizer = SymbolicMultitrackTokenizer()
    model = MultitrackCrossAttentionTransformer(tokenizer.vocab_sizes, d_model=128, nhead=8, num_layers=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = build_training_dataset()
    print(f"\n🧠 ENTRENANDO RED NEURONAL MULTITRACK EN EL DATASET ({len(dataset)} canciones procesadas)")
    print(f"  • Épocas objetivo: {epochs} | Optimizador: AdamW (lr={lr})")

    # Concatenar secuencias y crear batches de entrenamiento
    seq_len = 64
    chunks = []
    for seq in dataset:
        for i in range(0, len(seq) - seq_len, seq_len // 2):
            chunks.append(seq[i:i + seq_len])

    if not chunks:
        print("[ERROR] No se encontraron suficientes tokens para entrenar.")
        return

    X = torch.tensor(np.array(chunks), dtype=torch.long)
    print(f"  • Total chunks de entrenamiento generados: {X.shape[0]} (Largo: {seq_len})")

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Target: predecir el siguiente token
        input_tokens = X[:, :-1, :]
        target_tokens = X[:, 1:, :]

        out = model(input_tokens)

        # Pérdida multiobjetivo: Tono, Paso, Duración, Dinámica
        loss_pitch = criterion(out["pitch"].reshape(-1, 128), target_tokens[:, :, 1].reshape(-1))
        loss_step = criterion(out["step"].reshape(-1, 16), target_tokens[:, :, 2].reshape(-1))
        loss_dur = criterion(out["duration"].reshape(-1, 32), target_tokens[:, :, 3].reshape(-1))
        loss_vel = criterion(out["velocity"].reshape(-1, 8), target_tokens[:, :, 4].reshape(-1))

        total_loss = loss_pitch + loss_step + loss_dur + loss_vel
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [Época {epoch:2d}/{epochs}] Loss Total: {total_loss.item():.4f} (Pitch: {loss_pitch.item():.3f}, Step: {loss_step.item():.3f})")

    # Guardar pesos entrenados
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ ENTRENAMIENTO COMPLETADO. Pesos neuronales guardados en: {save_path}")

if __name__ == "__main__":
    train_cumbia_transformer(epochs=60)
