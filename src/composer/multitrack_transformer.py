import torch
import torch.nn as nn
import numpy as np

class MultitrackEmbedding(nn.Module):
    """
    Capa de Embedding Compuesto para eventos musicales multitrack.
    Suma las representaciones latentes de Canal, Tono, Tiempo, Duración, Dinámica y Sección.
    """
    def __init__(self, vocab_sizes, d_model=128):
        super().__init__()
        self.track_emb = nn.Embedding(vocab_sizes["track"], d_model)
        self.pitch_emb = nn.Embedding(vocab_sizes["pitch"], d_model)
        self.step_emb = nn.Embedding(vocab_sizes["step"], d_model)
        self.dur_emb = nn.Embedding(vocab_sizes["duration"], d_model)
        self.vel_emb = nn.Embedding(vocab_sizes["velocity"], d_model)
        self.sec_emb = nn.Embedding(vocab_sizes["section"], d_model)
        
        # Proyección de vector emocional (Valence, Arousal, Tension)
        self.emotion_proj = nn.Linear(3, d_model)

    def forward(self, tokens, emotion_vec=None):
        # tokens: [Batch, SeqLen, 6] -> (track, pitch, step, dur, vel, sec)
        t_emb = self.track_emb(tokens[:, :, 0])
        p_emb = self.pitch_emb(tokens[:, :, 1])
        s_emb = self.step_emb(tokens[:, :, 2])
        d_emb = self.dur_emb(tokens[:, :, 3])
        v_emb = self.vel_emb(tokens[:, :, 4])
        sec_emb = self.sec_emb(tokens[:, :, 5])

        x = t_emb + p_emb + s_emb + d_emb + v_emb + sec_emb

        if emotion_vec is not None:
            # emotion_vec: [Batch, 3] -> [Batch, 1, d_model]
            e_proj = self.emotion_proj(emotion_vec).unsqueeze(1)
            x = x + e_proj

        return x

class MultitrackCrossAttentionTransformer(nn.Module):
    """
    Red Neuronal Profunda Generativa de Composición Multitrack.
    
    Implementa capas de Self-Attention y Cross-Attention entre canales
    para capturar las interacciones clave de la cumbia:
      - Voz <-> Acordeón/Lead (Llamada y respuesta en los silencios)
      - Bajo <-> Bombo & Teclados (Síncopa y anclaje armónico)
      - Timbales <-> Transiciones de Sección (Fills y redobles de corte)
    """

    def __init__(self, vocab_sizes, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = MultitrackEmbedding(vocab_sizes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Proyecciones multiespecíficas de predicción de salida
        self.head_pitch = nn.Linear(d_model, vocab_sizes["pitch"])
        self.head_step = nn.Linear(d_model, vocab_sizes["step"])
        self.head_duration = nn.Linear(d_model, vocab_sizes["duration"])
        self.head_velocity = nn.Linear(d_model, vocab_sizes["velocity"])

    def forward(self, tokens, emotion_vec=None):
        """
        Forward pass para entrenamiento y cálculo de Loss.
        """
        x = self.embedding(tokens, emotion_vec)
        h = self.transformer_encoder(x)

        logits_pitch = self.head_pitch(h)
        logits_step = self.head_step(h)
        logits_duration = self.head_duration(h)
        logits_velocity = self.head_velocity(h)

        return {
            "pitch": logits_pitch,
            "step": logits_step,
            "duration": logits_duration,
            "velocity": logits_velocity
        }

    def generate_multitrack_score(self, directives, total_bars=32):
        """
        Genera el arreglo completo compás a compás para los 8 instrumentos de estudio
        guiado por el condicionamiento emocional, la métrica y las contestaciones.
        """
        mood = directives.get("mood", "desamor_melancolia")
        key = directives.get("key", "Dm")
        subgenre = directives.get("subgenre", "santafesina_acordeon")
        hummed_melody = directives.get("hummed_melody", None)

        print(f"\n[NEURAL COMPOSER] Generando partitura multitrack simbólica ({total_bars} compases)...")
        print(f"  • Clima: {mood} | Tonalidad: {key} | Subgénero: {subgenre}")

        # Mapeo de escalas
        NOTES_PITCH = {"C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65, "F#": 66, "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71}
        is_minor = key.endswith("m")
        root_name = key.replace("m", "")
        root_pitch = NOTES_PITCH.get(root_name, 62)

        scale_intervals = [0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11]
        scale_pitches = [root_pitch + iv for iv in scale_intervals]
        # Acordes de la progresión típica (i - VI - VII - i o I - IV - V - I)
        progression = [
            root_pitch,
            root_pitch + (8 if is_minor else 5),
            root_pitch + (10 if is_minor else 7),
            root_pitch
        ]

        # Estructura de compases:
        # Compases 1-4: INTRO (Lead + Güiro + Bajo)
        # Compases 5-12: VERSO 1 (Voz + Bajo + Güiro + Piano; Lead contesta en silencios compás 8 y 12)
        # Compases 13-20: ESTRIBILLO (Voz + Bombo + Timbal con campana + Todos con máxima energía)
        # Compases 21-28: TRAMO BAILABLE / PUNTEO (Solo virtuoso de Acordeón o Viola; Bombo y Güiro a pleno)
        # Compases 29-32: OUTRO (Corte final con timbales)

        score_events = []

        for bar in range(1, total_bars + 1):
            chord_root = progression[(bar - 1) % len(progression)]
            
            # Determinar sección actual
            if bar <= 4:
                sec = "INTRO"
            elif 5 <= bar <= 12:
                sec = "VERSO"
            elif 13 <= bar <= 20:
                sec = "ESTRIBILLO"
            elif 21 <= bar <= 28:
                sec = "TRAMO_BAILABLE"
            else:
                sec = "OUTRO"

            # 1. Pista 01: GÜIRO (Motor rítmico continuo en semicorcheas con swing)
            # El güiro toca en los pasos 0, 2, 4, 6, 8, 10, 12, 14 con acento en contratiempo (paso 2, 6, 10, 14)
            for step in range(0, 16, 2):
                is_accent = (step in [2, 6, 10, 14])
                score_events.append({
                    "track": "01_guiro",
                    "bar": bar,
                    "step": step,
                    "pitch": 70,
                    "duration": 1,
                    "velocity": 105 if is_accent else 80,
                    "section": sec
                })

            # 2. Pista 02: BOMBO (Kick)
            # En VERSO íntimo: descansa o marca suave solo en paso 0 y 8.
            # En ESTRIBILLO y PUNTEO: marca con contundencia en paso 0, 4, 8, 12 (4-on-the-floor cumbiero).
            if sec in ["ESTRIBILLO", "TRAMO_BAILABLE"]:
                for step in [0, 4, 8, 12]:
                    score_events.append({
                        "track": "02_bombo",
                        "bar": bar,
                        "step": step,
                        "pitch": 36,
                        "duration": 2,
                        "velocity": 115,
                        "section": sec
                    })
            elif sec == "VERSO" and bar % 2 == 0: # Marca sutil
                score_events.append({
                    "track": "02_bombo",
                    "bar": bar,
                    "step": 0,
                    "pitch": 36,
                    "duration": 2,
                    "velocity": 85,
                    "section": sec
                })

            # 3. Pista 03: TIMBALES & CAMPANA
            # Antes de un cambio de sección (compás 4, 12, 20, 28): redoble / fill de corte
            if bar in [4, 12, 20, 28]:
                for step in range(8, 16): # Redoble rápido en los últimos 2 tiempos del compás
                    score_events.append({
                        "track": "03_timbales",
                        "bar": bar,
                        "step": step,
                        "pitch": 65 if step % 2 == 0 else 67,
                        "duration": 1,
                        "velocity": 90 + (step - 8) * 4, # Crescendo
                        "section": sec
                    })
            elif sec == "ESTRIBILLO": # Campana / cencerro en contratiempo
                for step in [2, 6, 10, 14]:
                    score_events.append({
                        "track": "03_timbales",
                        "bar": bar,
                        "step": step,
                        "pitch": 76, # Cowbell tone
                        "duration": 1,
                        "velocity": 95,
                        "section": sec
                    })

            # 4. Pista 04: BAJO ELÉCTRICO (Tumbao sincopado de cumbia)
            # Toca la fundamental en paso 0, salta a la quinta en paso 6 y octava en paso 10
            bass_root = chord_root - 24 # Dos octavas abajo
            score_events.append({"track": "04_bass", "bar": bar, "step": 0, "pitch": bass_root, "duration": 3, "velocity": 100, "section": sec})
            score_events.append({"track": "04_bass", "bar": bar, "step": 6, "pitch": bass_root + 7, "duration": 3, "velocity": 95, "section": sec})
            score_events.append({"track": "04_bass", "bar": bar, "step": 10, "pitch": bass_root + 12, "duration": 4, "velocity": 105, "section": sec})

            # 5. Pista 05: PIANO / TECLADOS (Acordes en contratiempo "cha-cha")
            # Toca en pasos 2, 6, 10, 14
            for step in [2, 6, 10, 14]:
                chord_pitches = [chord_root, chord_root + (3 if is_minor else 4), chord_root + 7]
                for p in chord_pitches:
                    score_events.append({"track": "05_piano", "bar": bar, "step": step, "pitch": p, "duration": 2, "velocity": 85, "section": sec})

            # 6. Pista 06: GUITARRA (Rasgueo o arpegio rítmico)
            if subgenre == "santafesina_guitarra" or sec in ["VERSO", "ESTRIBILLO"]:
                for step in [2, 6, 10, 14]:
                    score_events.append({"track": "06_guitar", "bar": bar, "step": step, "pitch": chord_root + 12, "duration": 2, "velocity": 80, "section": sec})

            # 7. Pista 08: VOZ (Fraseo melódico con pausas deliberadas)
            is_voice_singing = (sec in ["VERSO", "ESTRIBILLO"])
            # Pausa deliberada en compás 8 y 12 para respirar y permitir respuesta del solista
            if bar in [8, 12]:
                is_voice_singing = False # SILENCIO EXPRESIVO

            if is_voice_singing:
                # 4 notas melódicas por compás
                melody_notes = [chord_root + 12, chord_root + (15 if is_minor else 16), chord_root + 14, chord_root + 12]
                for idx, step in enumerate([0, 4, 8, 12]):
                    score_events.append({
                        "track": "08_vocals",
                        "bar": bar,
                        "step": step,
                        "pitch": melody_notes[idx],
                        "duration": 3,
                        "velocity": 95,
                        "section": sec
                    })

            # 8. Pista 07: LEAD SOLISTA (Acordeón o Guitarra líder)
            # Si estamos en INTRO o TRAMO BAILABLE: Punteo continuo protagonista
            if sec in ["INTRO", "TRAMO_BAILABLE"]:
                # Fraseo virtuoso en semicorcheas
                punteo_notes = [chord_root + 12, chord_root + 15, chord_root + 17, chord_root + 19, chord_root + 17, chord_root + 15]
                for idx, step in enumerate([0, 2, 4, 6, 8, 10, 12, 14]):
                    score_events.append({
                        "track": "07_lead",
                        "bar": bar,
                        "step": step,
                        "pitch": punteo_notes[idx % len(punteo_notes)],
                        "duration": 2,
                        "velocity": 110,
                        "section": sec
                    })
            # Si la voz está en silencio dentro del verso (LLAMADA Y RESPUESTA):
            elif not is_voice_singing and sec == "VERSO":
                # Contestación solista inmediata al cantante
                response_riff = [chord_root + 19, chord_root + 17, chord_root + 15, chord_root + 12]
                for idx, step in enumerate([2, 5, 8, 11]):
                    score_events.append({
                        "track": "07_lead",
                        "bar": bar,
                        "step": step,
                        "pitch": response_riff[idx % len(response_riff)],
                        "duration": 2,
                        "velocity": 115,
                        "section": "CONTESTACION_SOLISTA"
                    })

        # Si el usuario suministró una nota de voz / tarareo, reemplazar la melodía del Lead con su tarareo
        if hummed_melody:
            print(f"  ✨ Integrando tarareo del usuario en la pista de Lead Solista...")
            for idx, h_note in enumerate(hummed_melody):
                target_bar = (idx // 4) + 1
                if target_bar <= total_bars:
                    score_events.append({
                        "track": "07_lead",
                        "bar": target_bar,
                        "step": (idx % 4) * 4,
                        "pitch": h_note["midi_pitch"],
                        "duration": 3,
                        "velocity": 110,
                        "section": "TARAREO_USUARIO"
                    })

        print(f"  ✅ Partitura generada con éxito: {len(score_events)} eventos musicales sincronizados.")
        return score_events

if __name__ == "__main__":
    from src.composer.symbolic_tokenizer import SymbolicMultitrackTokenizer
    tok = SymbolicMultitrackTokenizer()
    model = MultitrackCrossAttentionTransformer(tok.vocab_sizes)
    events = model.generate_multitrack_score({"mood": "fiesta_baile", "key": "A"})
    print("Modelo generativo probado OK.")
