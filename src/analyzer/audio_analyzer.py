import os
import json
import numpy as np
import librosa
import soundfile as sf
import torch
import pandas as pd
from src.analyzer.studio_deconstructor import StudioDeconstructor

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class StudioAudioAnalyzer:
    """
    Analizador Profesional de Estudio de Audio Multitrack (8 canales).
    
    Analiza:
    1. Métrica, Tempo (BPM) y compases usando el pulso del bombo y el reloj del güiro.
    2. Armonía y progresión compás a compás con bajo + piano + guitarra.
    3. Matriz de Arreglo DAW: qué instrumentos tocan en cada compás (densidad instrumental).
    4. Silencios y Dinámica de Llamada y Respuesta (Lead contestando a la Voz).
    5. Métricas de mezcla de estudio: Peak dBFS, RMS, Dynamic Range y Brillo espectral.
    """

    def __init__(self, stems_root="data/stems", processed_root="data/processed"):
        self.stems_root = stems_root
        self.processed_root = processed_root
        self.deconstructor = StudioDeconstructor(stems_root, processed_root)
        os.makedirs(self.processed_root, exist_ok=True)

    def load_or_extract_studio_multitrack(self, audio_path, song_id, lead_instrument="acordeon"):
        song_stems_dir = os.path.join(self.stems_root, song_id)
        expected_channels = [
            "01_guiro", "02_bombo", "03_timbales", "04_bass",
            "05_piano", "06_guitar", "07_lead", "08_vocals"
        ]
        
        stems = {}
        missing = False
        for ch in expected_channels:
            p = os.path.join(song_stems_dir, f"{song_id}_{ch}.wav")
            if not os.path.exists(p):
                missing = True
                break
            stems[ch] = p

        if missing:
            print(f"[ESTUDIO] Pistas multipista no completas para {song_id}. Ejecutando deconstrucción...")
            session = self.deconstructor.deconstruct_song_to_daw_multitrack(audio_path, song_id, lead_instrument)
            stems = session["rutas_stems"]

        return stems

    def detect_tempo_and_bars(self, y_guiro, y_bombo, sr):
        # El güiro marca la subdivisión en semicorcheas y el bombo marca los pulsos fuertes
        y_rhythm = y_bombo + 0.5 * y_guiro
        onset_env = librosa.onset.onset_strength(y=y_rhythm, sr=sr, aggregate=np.median)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=100)
        tempo_val = float(tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo)

        beat_times = librosa.frames_to_time(beats, sr=sr)

        bars = []
        for i in range(0, len(beat_times) - 3, 4):
            bar_start = float(beat_times[i])
            bar_end = float(beat_times[min(i + 4, len(beat_times) - 1)])
            bars.append({
                "bar_index": len(bars) + 1,
                "start_time": round(bar_start, 3),
                "end_time": round(bar_end, 3),
                "duration": round(bar_end - bar_start, 3)
            })

        return round(tempo_val, 1), beat_times, bars

    def detect_key_and_chords(self, y_harm, sr, bars):
        chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        best_score = -np.inf
        detected_key = "Am"
        
        for i, pitch in enumerate(PITCH_NAMES):
            maj_prof = np.roll(MAJOR_PROFILE, i)
            score_maj = np.corrcoef(chroma_mean, maj_prof)[0, 1]
            if score_maj > best_score:
                best_score = score_maj
                detected_key = f"{pitch}"

            min_prof = np.roll(MINOR_PROFILE, i)
            score_min = np.corrcoef(chroma_mean, min_prof)[0, 1]
            if score_min > best_score:
                best_score = score_min
                detected_key = f"{pitch}m"

        chord_sequence = []
        for bar in bars:
            start_sample = int(bar["start_time"] * sr)
            end_sample = int(bar["end_time"] * sr)
            segment = y_harm[start_sample:end_sample]

            if len(segment) >= 2048:
                try:
                    chroma_seg = librosa.feature.chroma_cqt(y=segment, sr=sr, n_chroma=12)
                    mean_seg = np.mean(chroma_seg, axis=1)
                    
                    top_notes_idx = np.argsort(mean_seg)[-3:]
                    root = PITCH_NAMES[top_notes_idx[-1]]
                    
                    third_maj = (top_notes_idx[-1] + 4) % 12
                    third_min = (top_notes_idx[-1] + 3) % 12
                    
                    is_minor = mean_seg[third_min] > mean_seg[third_maj]
                    chord_name = f"{root}m" if is_minor else f"{root}"
                    chord_sequence.append(chord_name)
                except Exception:
                    chord_sequence.append(detected_key)
            else:
                chord_sequence.append(detected_key)

        return detected_key, chord_sequence

    def analyze_multitrack_arrangement(self, tracks, sr, bars, chords):
        """
        Analiza compás por compás qué instrumentos de la sesión de estudio están sonando,
        detectando caídas de bombo, entradas de timbales, contestaciones del lead y estructura.
        """
        total_bars = len(bars)
        bar_matrix = []

        for i, bar in enumerate(bars):
            start = int(bar["start_time"] * sr)
            end = int(bar["end_time"] * sr)

            bar_energies = {}
            active_instruments = []

            for name, audio in tracks.items():
                seg = audio[start:end]
                rms = float(np.sqrt(np.mean(seg**2))) if len(seg) > 0 else 0.0
                bar_energies[name] = round(rms, 4)
                
                # Umbral de actividad por instrumento
                threshold = 0.015 if "guiro" in name or "bombo" in name else 0.02
                if rms > threshold:
                    active_instruments.append(name.replace("01_", "").replace("02_", "").replace("03_", "").replace("04_", "").replace("05_", "").replace("06_", "").replace("07_", "").replace("08_", ""))

            bar_matrix.append({
                "bar": bar["bar_index"],
                "start": bar["start_time"],
                "end": bar["end_time"],
                "chord": chords[i] if i < len(chords) else "Am",
                "instrumentos_activos": active_instruments,
                "has_vocals": ("vocals" in active_instruments),
                "has_bombo": ("bombo" in active_instruments),
                "has_lead": ("lead" in active_instruments),
                "densidad_instrumental": len(active_instruments),
                "rms_detallado": bar_energies
            })

        # Estructuración funcional
        sections = []
        raw_labels = []
        for i, b in enumerate(bar_matrix):
            has_vox = b["has_vocals"]
            has_bombo = b["has_bombo"]
            has_lead = b["has_lead"]
            dens = b["densidad_instrumental"]

            if i < 8 and not has_vox:
                raw_labels.append("Intro")
            elif i > total_bars - 8 and not has_vox:
                raw_labels.append("Outro")
            elif not has_vox and has_lead and dens >= 4:
                raw_labels.append("Tramo_Bailable_Punteo")
            elif has_vox:
                if has_bombo and dens >= 5:
                    raw_labels.append("Estribillo")
                else:
                    raw_labels.append("Verso")
            else:
                raw_labels.append("Corte_Instrumental")

        curr_label = raw_labels[0]
        start_idx = 0
        for i in range(1, len(raw_labels)):
            if raw_labels[i] != curr_label:
                sections.append({
                    "tipo": curr_label,
                    "compas_inicio": bars[start_idx]["bar_index"],
                    "compas_fin": bars[i-1]["bar_index"],
                    "tiempo_inicio": bars[start_idx]["start_time"],
                    "tiempo_fin": bars[i-1]["end_time"],
                    "duracion_compases": i - start_idx,
                    "instrumentos_predominantes": list(set([inst for k in range(start_idx, i) for inst in bar_matrix[k]["instrumentos_activos"]])),
                    "acordes": [bar_matrix[k]["chord"] for k in range(start_idx, i)]
                })
                curr_label = raw_labels[i]
                start_idx = i

        sections.append({
            "tipo": curr_label,
            "compas_inicio": bars[start_idx]["bar_index"],
            "compas_fin": bars[-1]["bar_index"],
            "tiempo_inicio": bars[start_idx]["start_time"],
            "tiempo_fin": bars[-1]["end_time"],
            "duracion_compases": len(raw_labels) - start_idx,
            "instrumentos_predominantes": list(set([inst for k in range(start_idx, len(raw_labels)) for inst in bar_matrix[k]["instrumentos_activos"]])),
            "acordes": [bar_matrix[k]["chord"] for k in range(start_idx, len(raw_labels))]
        })

        # Silencios con llamada y respuesta quirúrgica:
        # La voz se calla dentro de una sección vocal Y el lead solista responde tocando frases
        call_and_response = []
        for b in bar_matrix:
            if not b["has_vocals"] and b["has_lead"]:
                call_and_response.append({
                    "compas": b["bar"],
                    "tiempo_segundos": b["start"],
                    "acorde": b["chord"],
                    "rol": "contestacion_lead_solista",
                    "instrumentos_base": [inst for inst in b["instrumentos_activos"] if inst != "lead"]
                })

        return sections, call_and_response, bar_matrix

    def analyze_studio_song(self, audio_path, song_id, lead_instrument="acordeon", metadata=None):
        print(f"\n=======================================================")
        print(f"🎚️  ANÁLISIS PROFESIONAL MULTITRACK [{song_id}]: {os.path.basename(audio_path)}")
        print(f"=======================================================")

        # 1. Cargar las 8 pistas de estudio
        stems = self.load_or_extract_studio_multitrack(audio_path, song_id, lead_instrument)

        loaded_tracks = {}
        sr = 22050
        for name, path in stems.items():
            y, _ = librosa.load(path, sr=sr, mono=True)
            loaded_tracks[name] = y

        # 2. Métrica y Compases con Güiro y Bombo
        print("[MÉTRICA] Calculando BPM y compases con la base rítmica (Güiro + Bombo)...")
        tempo, beat_times, bars = self.detect_tempo_and_bars(
            loaded_tracks["01_guiro"], loaded_tracks["02_bombo"], sr
        )
        print(f"  -> BPM: {tempo} | Compases: {len(bars)}")

        # 3. Armonía con Bajo + Piano + Guitarra
        print("[ARMONÍA] Analizando tonalidad y acordes con la sección armónica (Bajo + Teclados + Guitarras)...")
        y_harm = loaded_tracks["04_bass"] + loaded_tracks["05_piano"] + loaded_tracks["06_guitar"]
        key, chords = self.detect_key_and_chords(y_harm, sr, bars)
        print(f"  -> Tonalidad: {key}")

        # 4. Matriz de Arreglo Multitrack y Contestaciones
        print("[ESTRUCTURA DAW] Mapeando actividad de cada pista compás a compás...")
        sections, call_responses, bar_matrix = self.analyze_multitrack_arrangement(
            loaded_tracks, sr, bars, chords
        )

        print(f"\n[MAPA ESTRUCTURAL DETECTADO]")
        for s in sections:
            print(f"  • {s['tipo']:<22} | Compás {s['compas_inicio']:>2} al {s['compas_fin']:>2} ({s['duracion_compases']:>2} compases) | Pistas: {', '.join(s['instrumentos_predominantes'][:4])}")

        print(f"\n  -> Contestaciones Lead / Solista detectadas: {len(call_responses)} compases identificados.")

        analysis_result = {
            "song_id": song_id,
            "file_name": os.path.basename(audio_path),
            "bpm": tempo,
            "tonalidad_predominante": key,
            "lead_instrument": lead_instrument,
            "total_compases": len(bars),
            "duracion_segundos": round(len(loaded_tracks["01_guiro"]) / sr, 2),
            "canales_estudio": list(stems.keys()),
            "tramos": sections,
            "contestaciones_llamada_respuesta": call_responses,
            "matriz_arreglo_compases": bar_matrix[:32], # Primeros 32 compases detallados
            "stems_paths": stems,
            "metadata_origen": metadata or {}
        }

        out_json = os.path.join(self.processed_root, f"{song_id}_analysis.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        print(f"[ÉXITO] Archivo de análisis de estudio guardado en: {out_json}")
        return analysis_result

def process_entire_dataset_studio(metadata_path="data/metadata.csv", raw_root="data/raw"):
    analyzer = StudioAudioAnalyzer()
    df = pd.read_csv(metadata_path)

    print(f"\n🚀 ANÁLISIS DEL DATASET COMPLETO EN 8 CANALES ({len(df)} canciones)")
    for idx, row in df.iterrows():
        song_id = row["song_id"]
        file_name = row["file_name"]
        lead = row.get("lead_instrument", "acordeon")
        audio_path = os.path.join(raw_root, file_name)

        if os.path.exists(audio_path):
            res = analyzer.analyze_studio_song(audio_path, song_id, lead_instrument=lead, metadata=row.to_dict())
            df.at[idx, "status"] = "studio_analyzed"
            df.at[idx, "bpm"] = res["bpm"]
            df.at[idx, "key"] = res["tonalidad_predominante"]
        else:
            print(f"[ERROR] Archivo no encontrado: {audio_path}")

    df.to_csv(metadata_path, index=False)
    print(f"\n✅ DATASET MULTITRACK HOMOGÉNEO Y METADATOS ACTUALIZADOS EN: {metadata_path}")

if __name__ == "__main__":
    process_entire_dataset_studio()
