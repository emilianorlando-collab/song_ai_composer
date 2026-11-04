import os
import subprocess
import numpy as np
import soundfile as sf
import librosa
from pedalboard import Pedalboard, Compressor, HighpassFilter, HighShelfFilter, PeakFilter, Reverb, Delay

class SingingVoiceEngine:
    """
    Motor de Síntesis Vocal Cantada y Expresiva de Estudio (Singing Voice Engine).
    
    Toma las letras generadas con su métrica poética y las convierte en una
    interpretación vocal afinada (cantada sobre la tonalidad de la canción):
      1. Generación fonética en español argentino nativo (Diego es_AR).
      2. Modulación y afinación tonal (Pitch-Shifting musical a las notas de la escala).
      3. Inflexiones vocales: Vibrato suave al final de los versos.
      4. Silencios y respiraciones sincronizados para la llamada y respuesta del acordeón.
      5. Cadena de procesamiento vocal de estudio: Compresor óptico, ecualización de presencia (3.5 kHz), aire (10 kHz) y Plate Reverb espaciosa.
    """

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

        # Cadena de procesamiento vocal de estudio (Vocal Channel Strip)
        self.vocal_strip = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=110.0), # Corte antipop
            Compressor(threshold_db=-18.0, ratio=3.5, attack_ms=15.0, release_ms=140.0),
            PeakFilter(cutoff_frequency_hz=3500.0, gain_db=3.0, q=1.2), # Presencia de voz
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=2.5),  # Brillo y aire
            Reverb(room_size=0.55, damping=0.4, wet_level=0.35, dry_level=0.85), # Plate Reverb
            Delay(delay_seconds=0.28, feedback=0.18, mix=0.20)         # Slapback delay sincronizado
        ])

    def render_sung_phrase(self, text, pitch_shift_semitones=2, add_vibrato=True):
        """
        Sintetiza una frase cantada afinada a la tonalidad musical de la canción.
        """
        tmp_aiff = f"/tmp/sing_phrase_{os.getpid()}.aiff"
        try:
            # Usar voz argentina con cadencia medida para canto
            cmd = ["say", "-v", "Diego", "-r", "150", text, "-o", tmp_aiff]
            subprocess.run(cmd, check=True)

            data, sr = sf.read(tmp_aiff)
            if sr != self.sr:
                data = librosa.resample(data.T, orig_sr=sr, target_sr=self.sr).T

            # Modular tono a la escala musical (Canto afinado / Auto-tune de estudio)
            if pitch_shift_semitones != 0:
                data = librosa.effects.pitch_shift(data, sr=self.sr, n_steps=pitch_shift_semitones)

            # Vibrato sutil en la cola de la frase
            if add_vibrato and len(data) > int(self.sr * 0.5):
                tail_len = int(self.sr * 0.6)
                t_tail = np.linspace(0, 0.6, tail_len)
                vib = 1.0 + 0.008 * np.sin(2 * np.pi * 5.2 * t_tail)
                data[-tail_len:] *= vib

            return data
        finally:
            if os.path.exists(tmp_aiff):
                os.remove(tmp_aiff)

    def generate_full_vocal_track(self, lyric_concept, key="Dm", bpm=105.0, total_bars=32):
        """
        Genera la pista vocal completa cantada para toda la canción,
        distribuyendo estrofas, estribillo y dejando silencios de respiración
        para que conteste el acordeón.
        """
        bar_dur = (60.0 / bpm) * 4.0
        total_samples = int(self.sr * total_bars * bar_dur)
        vocal_track = np.zeros(total_samples, dtype=np.float32)

        sec = lyric_concept["sections"]
        verso_1 = [l for l in sec["verso_1"] if not l.startswith("(")]
        estribillo = [l for l in sec["estribillo"] if not l.startswith("(")]
        verso_2 = [l for l in sec["verso_2"] if not l.startswith("(")]

        print(f"\n[VOCAL STUDIO] Produciendo pista vocal cantada afinada en {key}...")

        # Cronograma de compases con afinación melódica:
        # En Re menor (Dm), el verso canta en torno a Re4 / Fa4 (+2 semitonos)
        # El estribillo sube con fuerza y pasión a La4 / Si b4 (+5 a +7 semitonos)
        schedule = [
            # VERSO 1
            (5, verso_1[0] if len(verso_1) > 0 else "Esta noche entré al boliche sin pensar", 2),
            (6, verso_1[1] if len(verso_1) > 1 else "Con las ganas de apagar este dolor", 3),
            # COMPÁS 7 y 8: SILENCIO DE RESPIRACIÓN (Acordeón responde aquí)
            (9, verso_1[2] if len(verso_1) > 2 else "Pero al fondo te vi sonreír feliz", 2),
            (10, verso_1[3] if len(verso_1) > 3 else "Abrazada con el que era mi mejor amigo", 1),
            # COMPÁS 11 y 12: SILENCIO DE RESPIRACIÓN (Acordeón responde aquí)

            # ESTRIBILLO (Clímax emocional: tono alto con potencia y desgarro)
            (13, estribillo[0] if len(estribillo) > 0 else "¡Olvídala, me grita el corazón!", 6),
            (15, estribillo[1] if len(estribillo) > 1 else "¡Olvídala, que ya no tiene dueño tu perdón!", 7),
            (17, estribillo[2] if len(estribillo) > 2 else "Aunque me queme el pecho esta traición", 5),
            (19, estribillo[3] if len(estribillo) > 3 else "Seguiré bailando con mi desilusión", 4),

            # VERSO 2
            (29, verso_2[0] if len(verso_2) > 0 else "La botella ya se empieza a terminar", 2),
            (30, verso_2[1] if len(verso_2) > 1 else "Y el dolor se me clava más y más", 3)
            # COMPÁS 31 y 32: OUTRO
        ]

        for bar_idx, text_phrase, pitch_step in schedule:
            if bar_idx > total_bars:
                continue

            sung_data = self.render_sung_phrase(text_phrase, pitch_shift_semitones=pitch_step)
            start_sample = int((bar_idx - 1) * bar_dur * self.sr)
            end_sample = min(total_samples, start_sample + len(sung_data))
            copy_len = end_sample - start_sample

            # Fades en extremos para suavidad musical
            fade_len = min(int(self.sr * 0.04), copy_len // 4)
            if fade_len > 0:
                sung_data[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
                sung_data[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

            vocal_track[start_sample:end_sample] += sung_data[:copy_len]

        # Convertir a estéreo y procesar por el Channel Strip vocal de estudio
        vocal_stereo = np.stack([vocal_track, vocal_track], axis=0)
        vocal_processed = self.vocal_strip(vocal_stereo, self.sr)

        # Retornar formato [samples, 2]
        return vocal_processed.T

if __name__ == "__main__":
    from src.composer.sentiment_lyricist import SentimentLyricistEngine
    lyr = SentimentLyricistEngine()
    concept = lyr.generate_emotional_concept("desamor_desgarrador")
    engine = SingingVoiceEngine()
    out = engine.generate_full_vocal_track(concept, total_bars=8)
    print("Motor de canto probado OK. Shape:", out.shape)
