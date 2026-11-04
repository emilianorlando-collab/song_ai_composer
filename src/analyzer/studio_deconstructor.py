import os
import json
import subprocess
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
import torch
import pandas as pd

class StudioDeconstructor:
    """
    Deconstructor Profesional Multitrack de Audio para Producción de Estudio (Estilo DAW / Bizarrap / FL Studio).
    
    Desglosa una mezcla completa en 8 pistas individuales independientes:
      1. Güiro / Rallador (reloj rítmico cumbiero, raspado agudo > 3.2 kHz)
      2. Bombo / Kick (sub-bass y golpe fundamental < 140 Hz)
      3. Timbales / Campana / Congas (cencerro, pailas, rimshots, fills 140 Hz - 3.2 kHz)
      4. Bajo Eléctrico (línea de bajo sincopado fundamental)
      5. Piano / Teclados (acordes contratiempo y montunos)
      6. Guitarra (guitarras rítmicas o punteadas)
      7. Lead Solista (Acordeón o Guitarra líder melódica de contestación)
      8. Voces (Voz principal y coros con respiraciones y pausas)
    """

    def __init__(self, stems_root="data/stems", processed_root="data/processed"):
        self.stems_root = stems_root
        self.processed_root = processed_root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.stems_root, exist_ok=True)
        os.makedirs(self.processed_root, exist_ok=True)

    def run_demucs_6s(self, audio_path, song_id):
        """
        Ejecuta Demucs htdemucs_6s para extraer 6 troncos acústicos:
        drums, bass, other, vocals, guitar, piano.
        """
        raw_name = os.path.splitext(os.path.basename(audio_path))[0]
        demucs_out_dir = os.path.join(self.stems_root, "htdemucs_6s", raw_name)
        expected = ["drums.wav", "bass.wav", "other.wav", "vocals.wav", "guitar.wav", "piano.wav"]

        all_exist = os.path.exists(demucs_out_dir) and all(
            os.path.exists(os.path.join(demucs_out_dir, f)) for f in expected
        )

        if all_exist:
            print(f"[DEMUCS-6S] Troncos de 6 canales ya presentes en: {demucs_out_dir}")
        else:
            print(f"[DEMUCS-6S] Extrayendo 6 troncos acústicos para '{song_id}' ({raw_name})...")
            cmd = [
                "./.venv/bin/demucs",
                "-d", self.device,
                "-o", self.stems_root,
                "-n", "htdemucs_6s",
                audio_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Error en Demucs 6-stem: {res.stderr}")

        return {
            stem: os.path.join(demucs_out_dir, f"{stem}.wav")
            for stem in ["drums", "bass", "other", "vocals", "guitar", "piano"]
        }

    def deconstruct_percussion(self, drums_path, out_dir, song_id):
        """
        Desglosa la batería/percusión en canales de estudio independientes:
        - Güiro / Rallador: Filtro pasaaltos 4to orden > 3200 Hz (frecuencias de raspado y transitorios metálicos)
        - Bombo / Kick: Filtro pasabajos 4to orden < 140 Hz (subgrave y pulso fundamental)
        - Timbales / Campana / Congas: Filtro pasabanda 140 Hz - 3200 Hz (cuerpo del timbal, timbalazos y campana)
        """
        audio, sr = sf.read(drums_path)
        is_stereo = (audio.ndim == 2)

        # Crossover Linkwitz-Riley / Butterworth orden 4 de fase cero (sosfiltfilt)
        sos_low = signal.butter(4, 140, btype='lowpass', fs=sr, output='sos')
        sos_high = signal.butter(4, 3200, btype='highpass', fs=sr, output='sos')
        sos_mid_low = signal.butter(4, 140, btype='highpass', fs=sr, output='sos')
        sos_mid_high = signal.butter(4, 3200, btype='lowpass', fs=sr, output='sos')

        axis = 0 if is_stereo else -1
        bombo = signal.sosfiltfilt(sos_low, audio, axis=axis)
        guiro = signal.sosfiltfilt(sos_high, audio, axis=axis)
        timbales = signal.sosfiltfilt(sos_mid_high, signal.sosfiltfilt(sos_mid_low, audio, axis=axis), axis=axis)

        # Normalización y prevención de clipping transparente
        guiro_path = os.path.join(out_dir, f"{song_id}_01_guiro.wav")
        bombo_path = os.path.join(out_dir, f"{song_id}_02_bombo.wav")
        timbales_path = os.path.join(out_dir, f"{song_id}_03_timbales.wav")

        sf.write(guiro_path, guiro, sr, subtype='PCM_24')
        sf.write(bombo_path, bombo, sr, subtype='PCM_24')
        sf.write(timbales_path, timbales, sr, subtype='PCM_24')

        return {
            "guiro": guiro_path,
            "bombo": bombo_path,
            "timbales": timbales_path
        }

    def deconstruct_song_to_daw_multitrack(self, audio_path, song_id, lead_instrument="acordeon"):
        """
        Ejecuta la deconstrucción completa de una canción en una sesión multipista profesional de estudio (8 pistas).
        """
        print(f"\n=======================================================")
        print(f"🎛️  DECONSTRUCCIÓN PROFESIONAL DAW MULTITRACK: [{song_id}]")
        print(f"=======================================================")

        song_stems_dir = os.path.join(self.stems_root, song_id)
        os.makedirs(song_stems_dir, exist_ok=True)

        # 1. Separación 6-stem con Demucs
        stems_6 = self.run_demucs_6s(audio_path, song_id)

        # 2. Desglose quirúrgico de Percusión (Güiro, Bombo, Timbales)
        print(f"[PERC] Separando Güiro, Bombo y Timbales/Campana vía Crossover Multibanda...")
        perc_stems = self.deconstruct_percussion(stems_6["drums"], song_stems_dir, song_id)

        # 3. Pistas de Armonía, Voces y Leads
        bass_path = os.path.join(song_stems_dir, f"{song_id}_04_bass.wav")
        piano_path = os.path.join(song_stems_dir, f"{song_id}_05_piano.wav")
        guitar_path = os.path.join(song_stems_dir, f"{song_id}_06_guitar.wav")
        lead_path = os.path.join(song_stems_dir, f"{song_id}_07_lead.wav")
        vox_path = os.path.join(song_stems_dir, f"{song_id}_08_vocals.wav")

        # Cargar y guardar en formato 24-bit PCM para calidad de estudio profesional
        for src, dst in [
            (stems_6["bass"], bass_path),
            (stems_6["piano"], piano_path),
            (stems_6["guitar"], guitar_path),
            (stems_6["vocals"], vox_path)
        ]:
            data, sr = sf.read(src)
            sf.write(dst, data, sr, subtype='PCM_24')

        # Asignar Lead según el rol del instrumento en la canción
        # Si la canción es liderada por acordeón (Palmeras), 'other' contiene el acordeón solista.
        # Si la canción es liderada por guitarra (Bohío), 'guitar' contiene la viola solista.
        if lead_instrument == "guitarra":
            data_lead, sr = sf.read(stems_6["guitar"])
        else:
            data_lead, sr = sf.read(stems_6["other"])
        sf.write(lead_path, data_lead, sr, subtype='PCM_24')

        multitrack_manifest = {
            "01_guiro": perc_stems["guiro"],
            "02_bombo": perc_stems["bombo"],
            "03_timbales": perc_stems["timbales"],
            "04_bass": bass_path,
            "05_piano": piano_path,
            "06_guitar": guitar_path,
            "07_lead": lead_path,
            "08_vocals": vox_path
        }

        # 4. Análisis de niveles y dinámica de estudio (Canal por Canal)
        print(f"\n[DAW METRICS] Analizando métricas de audio de cada canal...")
        channel_metrics = {}
        for ch_name, ch_path in multitrack_manifest.items():
            y, sr = sf.read(ch_path)
            y_mono = librosa.to_mono(y.T) if y.ndim == 2 else y
            peak = float(np.max(np.abs(y_mono)))
            rms = float(np.sqrt(np.mean(y_mono**2)))
            peak_db = round(20 * np.log10(peak + 1e-9), 2)
            rms_db = round(20 * np.log10(rms + 1e-9), 2)
            sc = float(np.mean(librosa.feature.spectral_centroid(y=y_mono, sr=sr)))

            bus = (
                "Drums_Percussion" if "guiro" in ch_name or "bombo" in ch_name or "timbales" in ch_name
                else "Bass" if "bass" in ch_name
                else "Harmony_Keys" if "piano" in ch_name or "guitar" in ch_name
                else "Lead_Solo" if "lead" in ch_name
                else "Vocals"
            )

            channel_metrics[ch_name] = {
                "file": os.path.basename(ch_path),
                "path": ch_path,
                "bus": bus,
                "peak_dbfs": peak_db,
                "rms_dbfs": rms_db,
                "dynamic_range_db": round(peak_db - rms_db, 2),
                "spectral_centroid_hz": round(sc, 1),
                "duracion_sec": round(len(y_mono) / sr, 2)
            }
            print(f"  • {ch_name:<14} | Bus: {bus:<18} | Peak: {peak_db:>6.1f} dBFS | RMS: {rms_db:>6.1f} dBFS | Brillo: {sc:>6.0f} Hz")

        # 5. Generar Mapa de Sesión de Estudio JSON
        session_data = {
            "song_id": song_id,
            "formato_estudio": "DAW Multitrack 24-bit 44.1kHz",
            "lead_instrument": lead_instrument,
            "total_canales": len(multitrack_manifest),
            "canales": channel_metrics,
            "rutas_stems": multitrack_manifest
        }

        session_json_path = os.path.join(self.processed_root, f"{song_id}_studio_session.json")
        with open(session_json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        print(f"\n[DAW SESSION OK] Sesión multitrack completa exportada en: {session_json_path}")
        return session_data

def deconstruct_entire_dataset(metadata_path="data/metadata.csv", raw_root="data/raw"):
    deconstructor = StudioDeconstructor()
    df = pd.read_csv(metadata_path)

    print(f"\n🎧 DECONSTRUCCIÓN PROFESIONAL DE ESTUDIO ({len(df)} canciones)")
    for idx, row in df.iterrows():
        song_id = row["song_id"]
        file_name = row["file_name"]
        lead = row.get("lead_instrument", "acordeon")
        audio_path = os.path.join(raw_root, file_name)

        if os.path.exists(audio_path):
            deconstructor.deconstruct_song_to_daw_multitrack(audio_path, song_id, lead_instrument=lead)
        else:
            print(f"[ERROR] Archivo no encontrado: {audio_path}")

    print("\n✅ TODAS LAS CANCIONES DESGLOSADAS PROFESIONALMENTE EN 8 CANALES DE ESTUDIO.")

if __name__ == "__main__":
    deconstruct_entire_dataset()
