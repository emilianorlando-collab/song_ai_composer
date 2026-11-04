import os
import glob
import json
import time
import pandas as pd
import numpy as np
import torch
from src.analyzer.studio_deconstructor import StudioDeconstructor
from src.analyzer.audio_analyzer import StudioAudioAnalyzer
from src.composer.symbolic_tokenizer import SymbolicMultitrackTokenizer

class BatchDatasetProcessor:
    """
    Procesador Automatizado Masivo de Datasets de Audio para Entrenamiento Neuronal.
    
    Permite procesar decenas o cientos de canciones depositadas en 'data/raw/':
      1. Escaneo automático y detección de nuevos archivos (MP3, WAV, FLAC, M4A).
      2. Registro incremental en 'data/metadata.csv'.
      3. Deconstrucción profesional en 8 canales de estudio (24-bit PCM):
         - Güiro / Rallador
         - Bombo / Kick
         - Timbales & Campana
         - Bajo Eléctrico
         - Piano / Teclados
         - Guitarra
         - Lead Solista (Acordeón o Viola)
         - Voces
      4. Extracción de métricas de estudio: BPM, Tonalidad, Progresiones y Estructura.
      5. Consolidación en un Corpus Unificado de Entrenamiento PyTorch ('data/processed/unified_training_corpus.pt').
    """

    SUPPORTED_EXTENSIONS = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]

    def __init__(self, raw_dir="data/raw", stems_dir="data/stems", processed_dir="data/processed", metadata_path="data/metadata.csv"):
        self.raw_dir = raw_dir
        self.stems_dir = stems_dir
        self.processed_dir = processed_dir
        self.metadata_path = metadata_path

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.stems_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.deconstructor = StudioDeconstructor(stems_root=self.stems_dir, processed_root=self.processed_dir)
        self.analyzer = StudioAudioAnalyzer(stems_root=self.stems_dir, processed_root=self.processed_dir)
        self.tokenizer = SymbolicMultitrackTokenizer()

    def sync_metadata_catalog(self):
        """
        Escanea 'data/raw/' y sincroniza el archivo metadata.csv
        asegurando que cada canción tenga un ID único.
        """
        if os.path.exists(self.metadata_path):
            df = pd.read_csv(self.metadata_path)
        else:
            df = pd.DataFrame(columns=[
                "song_id", "file_name", "title", "artist", "subgenre",
                "bpm", "key", "mood", "lead_instrument", "structure_notes", "status"
            ])

        registered_files = set(df["file_name"].tolist())
        raw_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            raw_files.extend(glob.glob(os.path.join(self.raw_dir, f"*{ext}")))

        new_entries = []
        for fpath in sorted(raw_files):
            fname = os.path.basename(fpath)
            if fname not in registered_files:
                # Generar nuevo ID incremental
                next_idx = len(df) + len(new_entries) + 1
                new_id = f"cumbia_{next_idx:03d}"
                title_clean = os.path.splitext(fname)[0].replace("_", " ").title()

                new_entries.append({
                    "song_id": new_id,
                    "file_name": fname,
                    "title": title_clean,
                    "artist": "Artista Cumbia",
                    "subgenre": "santafesina_acordeon",
                    "bpm": 100.0,
                    "key": "Am",
                    "mood": "baile_fiesta",
                    "lead_instrument": "acordeon",
                    "structure_notes": "intro-verso-estribillo-solo-outro",
                    "status": "pending"
                })

        if new_entries:
            df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
            df.to_csv(self.metadata_path, index=False)
            print(f"[CATÁLOGO] Sincronizados {len(new_entries)} archivos nuevos en {self.metadata_path}.")
        else:
            print(f"[CATÁLOGO] Catálogo al día ({len(df)} canciones registradas).")

        return df

    def process_all_pending(self, force_reprocess=False):
        """
        Procesa de punta a punta todas las canciones pendientes del catálogo:
        Deconstrucción 8-stem -> Análisis de métrica y armonía -> Guardado estructurado.
        """
        df = self.sync_metadata_catalog()
        pending_rows = df[df["status"] != "studio_analyzed"] if not force_reprocess else df

        total_pending = len(pending_rows)
        print(f"\n=======================================================")
        print(f"🚀 INICIANDO PROCESAMIENTO MASIVO: {total_pending} canciones pendientes")
        print(f"=======================================================")

        if total_pending == 0:
            print("✅ Todas las canciones ya están deconstruidas y analizadas.")
            return df

        start_time = time.time()
        for idx, row in pending_rows.iterrows():
            song_id = row["song_id"]
            file_name = row["file_name"]
            lead_inst = row.get("lead_instrument", "acordeon")
            audio_path = os.path.join(self.raw_dir, file_name)

            if not os.path.exists(audio_path):
                print(f"⚠️ [SALTANDO] Archivo no encontrado en disco: {audio_path}")
                continue

            print(f"\n[{song_id}] Procesando {file_name}...")
            try:
                # 1. Deconstrucción a 8 canales
                session_data = self.deconstructor.deconstruct_song_to_daw_multitrack(
                    audio_path, song_id, lead_instrument=lead_inst
                )

                # 2. Análisis métrico, armónico y de silencios
                analysis_data = self.analyzer.analyze_studio_song(
                    audio_path, song_id, lead_instrument=lead_inst, metadata=row.to_dict()
                )

                # Actualizar estado en DataFrame
                df.at[idx, "status"] = "studio_analyzed"
                df.at[idx, "bpm"] = analysis_data["bpm"]
                df.at[idx, "key"] = analysis_data["tonalidad_predominante"]
                df.to_csv(self.metadata_path, index=False)

            except Exception as e:
                print(f"❌ [ERROR en {song_id}]: {e}")

        elapsed = round(time.time() - start_time, 2)
        print(f"\n✅ PROCESAMIENTO COMPLETADO EN {elapsed} SEGUNDOS.")
        return df

    def compile_unified_training_corpus(self, output_path="data/processed/unified_training_corpus.pt"):
        """
        Consolida todos los análisis JSON en un único corpus tensorizado en PyTorch
        para el entrenamiento masivo de la red neuronal holística.
        """
        print(f"\n[CORPUS] Compilando corpus unificado de entrenamiento...")
        json_files = glob.glob(os.path.join(self.processed_dir, "*_analysis.json"))
        
        all_token_matrices = []
        corpus_manifest = []

        for jf in sorted(json_files):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                song_id = data["song_id"]
                matriz = data.get("matriz_arreglo_compases", [])
                tramos = data.get("tramos", [])
                key = data.get("tonalidad_predominante", "Am")
                bpm = data.get("bpm", 100.0)

                events = []
                for bar in matriz:
                    bar_idx = bar["bar"]
                    active = bar.get("instrumentos_activos", [])
                    has_vox = bar.get("has_vocals", False)
                    has_bombo = bar.get("has_bombo", False)
                    has_lead = bar.get("has_lead", False)

                    sec = "VERSO"
                    for t in tramos:
                        if t["compas_inicio"] <= bar_idx <= t["compas_fin"]:
                            sec = t["tipo"]
                            break

                    # Güiro
                    for step in range(0, 16, 2):
                        events.append({"track": "01_guiro", "pitch": 70, "step": step, "duration": 1, "velocity": 90, "section": sec})
                    # Bombo
                    if has_bombo or sec in ["Estribillo", "Tramo_Bailable_Punteo"]:
                        for step in [0, 4, 8, 12]:
                            events.append({"track": "02_bombo", "pitch": 36, "step": step, "duration": 2, "velocity": 115, "section": sec})
                    # Timbales
                    if "timbales" in active:
                        for step in [2, 6, 10, 14]:
                            events.append({"track": "03_timbales", "pitch": 76, "step": step, "duration": 1, "velocity": 95, "section": sec})
                    # Bajo
                    events.append({"track": "04_bass", "pitch": 36, "step": 0, "duration": 3, "velocity": 100, "section": sec})
                    events.append({"track": "04_bass", "pitch": 43, "step": 6, "duration": 3, "velocity": 95, "section": sec})
                    events.append({"track": "04_bass", "pitch": 48, "step": 10, "duration": 4, "velocity": 105, "section": sec})
                    # Lead
                    if has_lead:
                        for step in [0, 2, 4, 6, 8, 10, 12, 14]:
                            events.append({"track": "07_lead", "pitch": 72, "step": step, "duration": 2, "velocity": 110, "section": sec})
                    elif not has_vox and sec == "Verso":
                        for step in [2, 5, 8, 11]:
                            events.append({"track": "07_lead", "pitch": 76, "step": step, "duration": 2, "velocity": 115, "section": "CONTESTACION"})
                    # Voz
                    if has_vox:
                        for step in [0, 4, 8, 12]:
                            events.append({"track": "08_vocals", "pitch": 67, "step": step, "duration": 3, "velocity": 95, "section": sec})

                tokens = self.tokenizer.tokenize_score(events)
                all_token_matrices.append(tokens)
                corpus_manifest.append({
                    "song_id": song_id,
                    "bpm": bpm,
                    "key": key,
                    "total_events": len(tokens)
                })

            except Exception as e:
                print(f"  [AVISO] No se pudo incorporar {jf}: {e}")

        # Guardar en PyTorch tensor
        torch.save({
            "token_matrices": all_token_matrices,
            "manifest": corpus_manifest
        }, output_path)

        manifest_path = output_path.replace(".pt", "_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(corpus_manifest, f, indent=2, ensure_ascii=False)

        print(f"✅ CORPUS UNIFICADO GUARDADO: {output_path} ({len(all_token_matrices)} obras consolidadas).")
        return output_path

if __name__ == "__main__":
    processor = BatchDatasetProcessor()
    processor.sync_metadata_catalog()
    processor.compile_unified_training_corpus()
