import json
import numpy as np

TRACK_NAMES = [
    "01_guiro", "02_bombo", "03_timbales", "04_bass",
    "05_piano", "06_guitar", "07_lead", "08_vocals"
]

SECTIONS = ["INTRO", "VERSO", "ESTRIBILLO", "TRAMO_BAILABLE", "OUTRO"]

class SymbolicMultitrackTokenizer:
    """
    Tokenizador Simbólico Multicanal para entrenamiento e inferencia en Deep Learning.
    
    Codifica eventos musicales multitrack en secuencias de vectores enteros
    para el Multitrack Cross-Attention Transformer en PyTorch.
    """

    def __init__(self, steps_per_bar=16):
        self.steps_per_bar = steps_per_bar # Semicorcheas en compás 4/4
        self.track_to_id = {t: i for i, t in enumerate(TRACK_NAMES)}
        self.id_to_track = {i: t for i, t in enumerate(TRACK_NAMES)}
        self.section_to_id = {s: i for i, s in enumerate(SECTIONS)}

        # Tamaños de vocabulario para embeddings en PyTorch
        self.vocab_sizes = {
            "track": len(TRACK_NAMES),
            "pitch": 128,              # 0 a 127 MIDI
            "step": self.steps_per_bar,# 0 a 15
            "duration": 32,            # 1 a 32 semicorcheas
            "velocity": 8,             # Cuantizado en 8 niveles dinámicos (ppp a fff)
            "section": len(SECTIONS),
            "special": 4               # [PAD], [BOS], [EOS], [CALL_SILENCE]
        }

    def quantize_velocity(self, vel):
        return int(np.clip(vel // 16, 0, 7))

    def encode_event(self, track_name, pitch, step, duration, velocity=80, section="VERSO"):
        track_id = self.track_to_id.get(track_name, 0)
        pitch_val = int(np.clip(pitch, 0, 127))
        step_val = int(step % self.steps_per_bar)
        dur_val = int(np.clip(duration, 1, 31))
        vel_val = self.quantize_velocity(velocity)
        sec_val = self.section_to_id.get(section, 1)

        return [track_id, pitch_val, step_val, dur_val, vel_val, sec_val]

    def decode_event(self, token_vector):
        track_id, pitch_val, step_val, dur_val, vel_val, sec_val = token_vector
        return {
            "track": self.id_to_track.get(int(track_id), "04_bass"),
            "pitch": int(pitch_val),
            "step": int(step_val),
            "duration": int(dur_val),
            "velocity": int(vel_val * 16 + 8),
            "section": SECTIONS[int(sec_val)] if int(sec_val) < len(SECTIONS) else "VERSO"
        }

    def tokenize_score(self, multitrack_events):
        """
        Convierte una lista de eventos estructurados en un tensor de tokens (N, 6).
        """
        tokens = []
        for ev in multitrack_events:
            vec = self.encode_event(
                track_name=ev["track"],
                pitch=ev.get("pitch", 60),
                step=ev.get("step", 0),
                duration=ev.get("duration", 2),
                velocity=ev.get("velocity", 90),
                section=ev.get("section", "VERSO")
            )
            tokens.append(vec)
        return np.array(tokens, dtype=np.int64)

if __name__ == "__main__":
    tokenizer = SymbolicMultitrackTokenizer()
    sample_ev = [{"track": "07_lead", "pitch": 69, "step": 4, "duration": 4, "velocity": 100, "section": "TRAMO_BAILABLE"}]
    tokens = tokenizer.tokenize_score(sample_ev)
    print("Tokenizador probado OK. Shape:", tokens.shape)
