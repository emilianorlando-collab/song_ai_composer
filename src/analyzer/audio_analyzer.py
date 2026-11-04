import os
import json
import subprocess
import numpy as np
import librosa
import soundfile as sf
import torch
import pandas as pd

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

class AudioAnalyzer:
    def __init__(self, stems_root="data/stems", processed_root="data/processed"):
        self.stems_root = stems_root
        self.processed_root = processed_root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.stems_root, exist_ok=True)
        os.makedirs(self.processed_root, exist_ok=True)

    def separate_stems(self, audio_path, song_id):
        """
        Separa la canción en 4 pistas (drums, bass, other, vocals) usando Demucs.
        Cada archivo WAV se nombra con el prefijo único del song_id para evitar confusiones.
        """
        song_stems_dir = os.path.join(self.stems_root, song_id)
        os.makedirs(song_stems_dir, exist_ok=True)

        expected_stems = {
            "drums": os.path.join(song_stems_dir, f"{song_id}_drums.wav"),
            "bass": os.path.join(song_stems_dir, f"{song_id}_bass.wav"),
            "other": os.path.join(song_stems_dir, f"{song_id}_other.wav"),
            "vocals": os.path.join(song_stems_dir, f"{song_id}_vocals.wav"),
        }

        # Soporte para migración de nombres viejos (drums.wav -> cumbia_002_drums.wav)
        for stem_name, new_path in expected_stems.items():
            old_path = os.path.join(song_stems_dir, f"{stem_name}.wav")
            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)

        # Si ya existen todos, no reprocesamos
        if all(os.path.exists(p) for p in expected_stems.values()):
            print(f"[STEMS] Pistas ya separadas en: {song_stems_dir}")
            return expected_stems

        print(f"\n[DEMUCS] Separando stems para {song_id} ({os.path.basename(audio_path)})...")
        cmd = [
            "./.venv/bin/demucs",
            "-d", self.device,
            "-o", self.stems_root,
            "-n", "htdemucs",
            audio_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Error en Demucs: {res.stderr}")

        raw_name = os.path.splitext(os.path.basename(audio_path))[0]
        demucs_out_dir = os.path.join(self.stems_root, "htdemucs", raw_name)
        
        stems_dict = {}
        for stem_name in ["drums", "bass", "other", "vocals"]:
            src = os.path.join(demucs_out_dir, f"{stem_name}.wav")
            dst = expected_stems[stem_name]
            if os.path.exists(src):
                os.replace(src, dst)
                stems_dict[stem_name] = dst
            else:
                stems_dict[stem_name] = src

        # Limpiar carpeta temporal de Demucs si quedó vacía
        try:
            if os.path.exists(demucs_out_dir) and not os.listdir(demucs_out_dir):
                os.rmdir(demucs_out_dir)
        except Exception:
            pass

        print(f"[STEMS OK] Pistas guardadas con ID '{song_id}' en: {song_stems_dir}")
        return stems_dict

    def detect_tempo_and_bars(self, y_drums, sr):
        onset_env = librosa.onset.onset_strength(y=y_drums, sr=sr, aggregate=np.median)
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

    def detect_key_and_chords(self, y_other, y_bass, sr, bars):
        y_harm = y_other + y_bass
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

    def segment_structure_and_silences(self, y_vocals, y_drums, y_other, sr, bars, chords):
        total_bars = len(bars)
        bar_stats = []

        for i, bar in enumerate(bars):
            start = int(bar["start_time"] * sr)
            end = int(bar["end_time"] * sr)

            v_seg = y_vocals[start:end]
            d_seg = y_drums[start:end]
            o_seg = y_other[start:end]

            rms_v = float(np.sqrt(np.mean(v_seg**2))) if len(v_seg) > 0 else 0.0
            rms_d = float(np.sqrt(np.mean(d_seg**2))) if len(d_seg) > 0 else 0.0
            rms_o = float(np.sqrt(np.mean(o_seg**2))) if len(o_seg) > 0 else 0.0

            bar_stats.append({
                "bar": bar["bar_index"],
                "start": bar["start_time"],
                "end": bar["end_time"],
                "rms_v": rms_v,
                "rms_d": rms_d,
                "rms_o": rms_o,
                "has_voice": bool(rms_v > 0.035),
                "chord": chords[i] if i < len(chords) else "Am"
            })

        # Suavizado temporal
        smoothed_voice = []
        for i in range(total_bars):
            window = [bar_stats[j]["has_voice"] for j in range(max(0, i-1), min(total_bars, i+2))]
            smoothed_voice.append(sum(window) >= 2)

        raw_labels = []
        for i, b in enumerate(bar_stats):
            voice_active = smoothed_voice[i]
            drums_energy = b["rms_d"]
            other_energy = b["rms_o"]

            if i < 8 and not voice_active:
                raw_labels.append("Intro")
            elif i > total_bars - 6 and not voice_active:
                raw_labels.append("Outro")
            elif not voice_active and (other_energy > 0.035 or drums_energy > 0.07):
                raw_labels.append("Tramo_Bailable_Punteo")
            elif voice_active:
                if drums_energy > 0.11:
                    raw_labels.append("Estribillo")
                else:
                    raw_labels.append("Verso")
            else:
                raw_labels.append("Puente_Instrumental")

        sections = []
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
                    "presencia_voz": any(bar_stats[k]["has_voice"] for k in range(start_idx, i)),
                    "acordes": [bar_stats[k]["chord"] for k in range(start_idx, i)]
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
            "presencia_voz": any(bar_stats[k]["has_voice"] for k in range(start_idx, len(raw_labels))),
            "acordes": [bar_stats[k]["chord"] for k in range(start_idx, len(raw_labels))]
        })

        call_and_response_silences = []
        for sec in sections:
            if sec["presencia_voz"]:
                for b in bar_stats:
                    if sec["compas_inicio"] <= b["bar"] <= sec["compas_fin"]:
                        if not b["has_voice"] and b["rms_o"] > 0.03:
                            call_and_response_silences.append({
                                "compas": b["bar"],
                                "tiempo_segundos": b["start"],
                                "acorde": b["chord"],
                                "en_seccion": sec["tipo"],
                                "oportunidad_respuesta_solista": True
                            })

        return sections, call_and_response_silences

    def analyze_song(self, audio_path, song_id, metadata=None):
        print(f"\n=======================================================")
        print(f"🎵 PROCESANDO CANCIÓN [{song_id}]: {os.path.basename(audio_path)}")
        print(f"=======================================================")

        stems = self.separate_stems(audio_path, song_id)

        y_drums, sr = librosa.load(stems["drums"], sr=22050, mono=True)
        y_bass, _ = librosa.load(stems["bass"], sr=22050, mono=True)
        y_other, _ = librosa.load(stems["other"], sr=22050, mono=True)
        y_vox, _ = librosa.load(stems["vocals"], sr=22050, mono=True)

        print("\n[MÉTRICA] Calculando tempo y compases (downbeats)...")
        tempo, beat_times, bars = self.detect_tempo_and_bars(y_drums, sr)
        print(f"  -> BPM: {tempo} | Compases: {len(bars)}")

        print("\n[ARMONÍA] Analizando tonalidad y progresiones...")
        key, chords = self.detect_key_and_chords(y_other, y_bass, sr, bars)
        print(f"  -> Tonalidad: {key}")

        print("\n[ESTRUCTURA] Segmentando tramos funcionales y silencios...")
        sections, call_responses = self.segment_structure_and_silences(y_vox, y_drums, y_other, sr, bars, chords)

        print(f"\n[MAPA ESTRUCTURAL DETECTADO]")
        for s in sections:
            print(f"  • {s['tipo']:<22} | Compás {s['compas_inicio']:>2} al {s['compas_fin']:>2} ({s['duracion_compases']:>2} compases) | Acordes: {' - '.join(s['acordes'][:4])}")

        print(f"\n  -> Silencios para llamada y respuesta: {len(call_responses)} compases identificados.")

        analysis_result = {
            "song_id": song_id,
            "file_name": os.path.basename(audio_path),
            "bpm": tempo,
            "tonalidad_predominante": key,
            "total_compases": len(bars),
            "duracion_segundos": round(len(y_drums) / sr, 2),
            "tramos": sections,
            "silencios_llamada_respuesta": call_responses,
            "stems_paths": stems,
            "metadata_origen": metadata or {}
        }

        out_json = os.path.join(self.processed_root, f"{song_id}_analysis.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        print(f"\n[ÉXITO] Archivo estructurado guardado en: {out_json}")
        return analysis_result

def process_entire_dataset(metadata_path="data/metadata.csv", raw_root="data/raw"):
    analyzer = AudioAnalyzer()
    df = pd.read_csv(metadata_path)
    
    print(f"\n🚀 PROCESANDO DATASET COMPLETO ({len(df)} canciones)")
    
    for idx, row in df.iterrows():
        song_id = row["song_id"]
        file_name = row["file_name"]
        audio_path = os.path.join(raw_root, file_name)
        
        if os.path.exists(audio_path):
            res = analyzer.analyze_song(audio_path, song_id, row.to_dict())
            df.at[idx, "status"] = "analyzed"
            df.at[idx, "bpm"] = res["bpm"]
            df.at[idx, "key"] = res["tonalidad_predominante"]
        else:
            print(f"[ERROR] Archivo no encontrado: {audio_path}")
            
    df.to_csv(metadata_path, index=False)
    print(f"\n✅ DATASET HOMOGÉNEO Y METADATOS ACTUALIZADOS EN: {metadata_path}")

if __name__ == "__main__":
    process_entire_dataset()
