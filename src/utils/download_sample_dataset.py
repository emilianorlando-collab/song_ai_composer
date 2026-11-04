import os
import subprocess
import imageio_ffmpeg
import pandas as pd

RAW_DIR = "/Volumes/T7/song_ai_composer/data/raw"
METADATA_FILE = "/Volumes/T7/song_ai_composer/data/metadata.csv"
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

SONGS = [
    {
        "id": "cumbia_001",
        "file_name": "cumbia_001_los_palmeras_olvidala.mp3",
        "query": "ytsearch1:Los Palmeras Olvidala audio oficial",
        "title": "Olvídala",
        "artist": "Los Palmeras",
        "subgenre": "santafesina_romantica",
        "bpm": 100,
        "key": "Am",
        "mood": "desamor_melancolia",
        "lead_instrument": "acordeon",
        "structure_notes": "intro-verso-estribillo-punteo-outro",
    },
    {
        "id": "cumbia_002",
        "file_name": "cumbia_002_los_palmeras_el_bombon_asesino.mp3",
        "query": "ytsearch1:Los Palmeras El Bombon Asesino audio oficial",
        "title": "El Bombón Asesino",
        "artist": "Los Palmeras",
        "subgenre": "santafesina_bailable",
        "bpm": 102,
        "key": "Am",
        "mood": "fiesta_baile",
        "lead_instrument": "acordeon",
        "structure_notes": "intro-verso-estribillo-corte_timbal-outro",
    },
    {
        "id": "cumbia_003",
        "file_name": "cumbia_003_los_del_bohio_vete_ya.mp3",
        "query": "ytsearch1:Los del Bohio Vete Ya audio oficial",
        "title": "Vete Ya",
        "artist": "Los del Bohío",
        "subgenre": "santafesina_guitarra",
        "bpm": 98,
        "key": "Dm",
        "mood": "nostalgia_desamor",
        "lead_instrument": "guitarra",
        "structure_notes": "intro-verso-estribillo-solo_guitarra-outro",
    }
]

def download_songs():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    metadata_rows = []
    
    for s in SONGS:
        out_path = os.path.join(RAW_DIR, s["file_name"])
        base_name = out_path.rsplit(".", 1)[0]
        
        if os.path.exists(out_path):
            print(f"[SKIP] {s['file_name']} ya existe.")
        else:
            print(f"\n[DESCARGANDO] {s['artist']} - {s['title']}...")
            cmd = [
                "/Volumes/T7/song_ai_composer/.venv/bin/yt-dlp",
                "--ffmpeg-location", FFMPEG_PATH,
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", f"{base_name}.%(ext)s",
                s["query"]
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"[OK] Guardado en: {out_path}")
            else:
                print(f"[ERROR] Error al descargar {s['title']}: {res.stderr}")
                
        metadata_rows.append({
            "song_id": s["id"],
            "file_name": s["file_name"],
            "title": s["title"],
            "artist": s["artist"],
            "subgenre": s["subgenre"],
            "bpm": s["bpm"],
            "key": s["key"],
            "mood": s["mood"],
            "lead_instrument": s["lead_instrument"],
            "structure_notes": s["structure_notes"],
            "status": "raw"
        })
        
    df = pd.DataFrame(metadata_rows)
    df.to_csv(METADATA_FILE, index=False)
    print(f"\n[METADATA] Actualizado {METADATA_FILE}")

if __name__ == "__main__":
    download_songs()
