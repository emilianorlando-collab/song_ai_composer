import os
import numpy as np
import librosa
import soundfile as sf

class MultimodalInputHandler:
    """
    Manejador de Entradas Multimodales para Song AI Composer.
    
    Admite 4 modalidades de entrada:
    1. Vacía (Zero-shot): La IA decide tema, emoción, tempo y letra de forma autónoma.
    2. Prompt Semántico / Emocional: Texto con el sentimiento deseado (ej: 'Cumbia melancólica de desamor').
    3. Letra Personalizada del Usuario: Análisis métrico, silábico, acentos y asignación a compases con pausas.
    4. Audio / Nota de Voz / Tarareo (WhatsApp memo): Extracción de tono F0 con pYIN para convertir la voz cantada o tarareada en notas musicales cuantizadas.
    """

    def __init__(self, sample_rate=22050):
        self.sr = sample_rate

    def parse_humming_voice_memo(self, audio_path, hop_length=512, target_bpm=None):
        """
        Toma una nota de voz de WhatsApp o grabación tarareada (WAV/MP3/M4A/OGG),
        extrae la trayectoria de tono fundamental (F0) usando pYIN y la convierte
        en una secuencia de notas cuantizadas con onset y duración.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio no encontrado: {audio_path}")

        print(f"\n[MULTIMODAL] Procesando nota de voz/tarareo: {os.path.basename(audio_path)}...")
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Filtro pasaaltos para eliminar ruidos de viento / manipulación del celular
        y = librosa.effects.preemphasis(y)

        # Estimación probabilística de F0 con pYIN (rango de voz humana: 65 Hz a 600 Hz)
        fmin = librosa.note_to_hz('C2')
        fmax = librosa.note_to_hz('C6')
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
        )

        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

        # Segmentar en notas discretas
        notes = []
        current_note = None
        start_time = 0.0

        for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
            t = float(times[i])
            if is_voiced and freq is not None and not np.isnan(freq):
                midi_pitch = int(round(librosa.hz_to_midi(freq)))
                if current_note is None:
                    current_note = midi_pitch
                    start_time = t
                elif abs(midi_pitch - current_note) >= 1.0: # Cambio de nota
                    dur = t - start_time
                    if dur >= 0.10: # Duración mínima de 100ms para evitar transitorios falsos
                        notes.append({
                            "midi_pitch": int(current_note),
                            "note_name": librosa.midi_to_note(current_note),
                            "start_time": round(start_time, 3),
                            "duration": round(dur, 3)
                        })
                    current_note = midi_pitch
                    start_time = t
            else:
                if current_note is not None:
                    dur = t - start_time
                    if dur >= 0.10:
                        notes.append({
                            "midi_pitch": int(current_note),
                            "note_name": librosa.midi_to_note(current_note),
                            "start_time": round(start_time, 3),
                            "duration": round(dur, 3)
                        })
                    current_note = None

        print(f"  -> Notas extraídas del tarareo: {len(notes)} eventos melódicos identificados.")
        return notes

    def parse_user_lyrics(self, lyrics_text):
        """
        Analiza la letra ingresada por el usuario:
        - Divide en estrofas y versos.
        - Calcula sílabas aproximadas por verso.
        - Identifica puntos de respiración / silencios deliberados para que la voz no sature.
        """
        lines = [line.strip() for line in lyrics_text.strip().split("\n") if line.strip()]
        stanzas = []
        current_stanza = []

        for line in lines:
            if line.startswith("[") and line.endswith("]"): # Etiqueta de sección como [Verso], [Coro]
                if current_stanza:
                    stanzas.append(current_stanza)
                    current_stanza = []
            words = line.split()
            # Estimación de sílabas en español (vocales + diptongos)
            vowels = "aeiouáéíóúü"
            syllables = sum(sum(1 for char in word.lower() if char in vowels) for word in words)
            syllables = max(syllables, 1)

            current_stanza.append({
                "line": line,
                "words": words,
                "syllable_count": syllables,
                "requires_breath_silence": (len(words) <= 5 or line.endswith((".", ",", "!", "?")))
            })

        if current_stanza:
            stanzas.append(current_stanza)

        print(f"[MULTIMODAL] Letra analizada: {len(lines)} versos distribuidos en {len(stanzas)} secciones.")
        return stanzas

    def resolve_song_directives(self, prompt=None, lyrics_text=None, audio_memo_path=None, mood=None, bpm=None, key=None):
        """
        Resuelve y unifica todas las modalidades de entrada en una especificación
        completa para el generador de canciones.
        """
        resolved = {
            "title": "Cumbia Generativa de Estudio",
            "mood": mood or "desamor_melancolia",
            "bpm": bpm or (68.0 if mood == "desamor_melancolia" else 102.0),
            "key": key or ("Dm" if mood == "desamor_melancolia" else "A"),
            "subgenre": "santafesina_acordeon",
            "lyrics": None,
            "hummed_melody": None,
            "input_mode": "zero_shot"
        }

        # 1. Si hay tarareo
        if audio_memo_path and os.path.exists(audio_memo_path):
            resolved["hummed_melody"] = self.parse_humming_voice_memo(audio_memo_path)
            resolved["input_mode"] = "audio_memo_guided"

        # 2. Si hay letra
        if lyrics_text and lyrics_text.strip():
            resolved["lyrics"] = self.parse_user_lyrics(lyrics_text)
            resolved["input_mode"] = "lyrics_guided" if resolved["input_mode"] == "zero_shot" else "multimodal_full"

        # 3. Si hay prompt de texto
        if prompt:
            prompt_lower = prompt.lower()
            if any(w in prompt_lower for w in ["fiesta", "baile", "alegre", "parranda"]):
                resolved["mood"] = "fiesta_baile"
                resolved["bpm"] = bpm or 105.0
                resolved["key"] = key or "A"
            elif any(w in prompt_lower for w in ["guitarra", "bohio", "lady", "tristeza"]):
                resolved["subgenre"] = "santafesina_guitarra"
                resolved["mood"] = "nostalgia_desamor"
                resolved["bpm"] = bpm or 95.0
                resolved["key"] = key or "Em"
            elif any(w in prompt_lower for w in ["olvidala", "desamor", "llorar", "romantica"]):
                resolved["mood"] = "desamor_melancolia"
                resolved["bpm"] = bpm or 66.0
                resolved["key"] = key or "Dm"

        print(f"\n[DIRECTIVAS DE ESTUDIO]")
        print(f"  • Modalidad: {resolved['input_mode']}")
        print(f"  • Sentimiento: {resolved['mood']} | Tonalidad: {resolved['key']} | Tempo: {resolved['bpm']} BPM")
        print(f"  • Subgénero: {resolved['subgenre']}")
        return resolved

if __name__ == "__main__":
    handler = MultimodalInputHandler()
    res = handler.resolve_song_directives(prompt="Quiero una cumbia de fiesta bailable bien santafesina")
    print("Directivas de prueba resueltas OK.")
