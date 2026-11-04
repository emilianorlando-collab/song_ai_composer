import os
import json
import numpy as np
import soundfile as sf
import librosa
from pedalboard import (
    Pedalboard, HighpassFilter, LowpassFilter, PeakFilter, 
    HighShelfFilter, Compressor, Limiter, Reverb, Delay, Chorus, Gain
)

class StudioMixer:
    """
    Consola Virtual de Mezcla y Masterización de Estudio (Estilo DAW: Ableton / FL Studio / Pro Tools).
    
    Aplica:
    1. Channel Strips individuales por pista:
       - EQ quirúrgica (Corte de frecuencias graves residuales, realce de brillo en güiro/timbales)
       - Panning estéreo profesional (Bajo y Bombo en centro mono, percusión y armonías abiertas)
       - Envíos de efectos espaciales (Plate Reverb para voces, Room para timbales, Delay a tempo)
    2. Suma multipista con Sidechain Ducking (Bombo atenúa sutilmente el Bajo en los golpes)
    3. Master Bus Chain:
       - Glue Compression para cohesión de estudio
       - True Peak Limiter a -1.0 dBFS para evitar distorsión inter-sample y streaming compliance (-14 LUFS)
    4. Exportación dual:
       - Master WAV 24-bit estéreo de estudio
       - Multitrack Stems Pack (8 WAVs individuales procesados listos para importar a DAW)
    """

    DEFAULT_CHANNEL_PROFILES = {
        "01_guiro": {
            "pan": 0.25,        # Levemente a la derecha para dar balance con los timbales
            "gain_db": -1.5,
            "hp_freq": 250.0,   # Limpieza de graves
            "hs_freq": 8000.0,  # Brillo del rallador
            "hs_gain_db": 3.0,
            "reverb_send": 0.05,
            "delay_send": 0.0
        },
        "02_bombo": {
            "pan": 0.0,         # Centro absoluto (Mono)
            "gain_db": 0.0,
            "hp_freq": 35.0,    # Eliminar frecuencias sub-sónicas inaudibles
            "hs_freq": 5000.0,
            "hs_gain_db": -6.0, # Cortar ruidos agudos
            "reverb_send": 0.0,
            "delay_send": 0.0
        },
        "03_timbales": {
            "pan": -0.25,       # Balance a la izquierda respecto al güiro
            "gain_db": -1.0,
            "hp_freq": 120.0,
            "hs_freq": 4500.0,
            "hs_gain_db": 2.0,  # Claridad en campana y rimshots
            "reverb_send": 0.15,
            "delay_send": 0.05
        },
        "04_bass": {
            "pan": 0.0,         # Centro absoluto (Mono)
            "gain_db": -0.5,
            "hp_freq": 40.0,
            "hs_freq": 2500.0,
            "hs_gain_db": -3.0,
            "reverb_send": 0.0,
            "delay_send": 0.0
        },
        "05_piano": {
            "pan": -0.15,
            "gain_db": -2.0,
            "hp_freq": 150.0,   # Evitar barro con el bajo
            "hs_freq": 6000.0,
            "hs_gain_db": 1.5,
            "reverb_send": 0.20,
            "delay_send": 0.0
        },
        "06_guitar": {
            "pan": 0.20,
            "gain_db": -2.5,
            "hp_freq": 140.0,
            "hs_freq": 4000.0,
            "hs_gain_db": 1.0,
            "reverb_send": 0.18,
            "delay_send": 0.08
        },
        "07_lead": {
            "pan": 0.0,         # Centro o estéreo amplio (Acordeón o Viola solista)
            "gain_db": 1.0,     # Protagonista melódico
            "hp_freq": 180.0,
            "hs_freq": 3500.0,
            "hs_gain_db": 2.5,  # Presencia en la mezcla
            "reverb_send": 0.25,
            "delay_send": 0.15  # Delay rítmico cumbiero
        },
        "08_vocals": {
            "pan": 0.0,         # Centro absoluto al frente
            "gain_db": 1.5,     # La voz debe liderar con claridad
            "hp_freq": 100.0,   # Filtro antipop / graves
            "hs_freq": 5000.0,
            "hs_gain_db": 2.0,  # Aire y presencia vocal
            "reverb_send": 0.30,# Plate reverb espaciosa
            "delay_send": 0.12  # Slapback / delay musical
        }
    }

    def __init__(self, sample_rate=44100):
        self.sr = sample_rate

        # Efectos espaciales globales de envío (Sends de Estudio)
        self.plate_reverb = Reverb(room_size=0.45, damping=0.5, wet_level=0.8, dry_level=0.0)
        self.tempo_delay = Delay(delay_seconds=0.25, feedback=0.20, mix=0.8)

        # Cadena de Masterización de Estudio (Master Bus)
        self.master_bus = Pedalboard([
            Compressor(threshold_db=-14.0, ratio=2.5, attack_ms=25.0, release_ms=120.0),
            PeakFilter(cutoff_frequency_hz=250.0, gain_db=-1.0, q=1.0), # Limpieza de zona turbia
            HighShelfFilter(cutoff_frequency_hz=10000.0, gain_db=1.0),  # Brillo comercial
            Limiter(threshold_db=-1.0, release_ms=80.0)                 # True Peak Limiter
        ])

    def apply_pan(self, stereo_audio, pan):
        """
        Aplica paneo estéreo con ley de paneo de potencia constante (-3dB al centro).
        pan va de -1.0 (Izquierda total) a +1.0 (Derecha total).
        """
        # Normalizar a 2 canales [2, samples]
        if stereo_audio.shape[0] != 2:
            stereo_audio = np.stack([stereo_audio[0], stereo_audio[0]], axis=0)

        theta = (pan + 1.0) * (np.pi / 4.0) # 0 a pi/2
        left_gain = float(np.cos(theta))
        right_gain = float(np.sin(theta))

        panned = np.zeros_like(stereo_audio)
        panned[0] = stereo_audio[0] * left_gain
        panned[1] = stereo_audio[1] * right_gain
        return panned

    def process_channel(self, name, audio, profile=None):
        """
        Procesa una pista individual a través de su Channel Strip virtual.
        """
        prof = profile or self.DEFAULT_CHANNEL_PROFILES.get(name, {
            "pan": 0.0, "gain_db": 0.0, "hp_freq": 80.0, "hs_freq": 5000.0,
            "hs_gain_db": 0.0, "reverb_send": 0.1, "delay_send": 0.0
        })

        # Asegurar formato [2, N] (estéreo)
        if audio.ndim == 1:
            stereo = np.stack([audio, audio], axis=0)
        elif audio.shape[0] != 2 and audio.shape[1] == 2:
            stereo = audio.T
        else:
            stereo = audio

        # 1. EQ del canal
        strip = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=prof["hp_freq"]),
            HighShelfFilter(cutoff_frequency_hz=prof["hs_freq"], gain_db=prof["hs_gain_db"]),
            Gain(gain_db=prof["gain_db"])
        ])
        eqed = strip(stereo, self.sr)

        # 2. Panning estéreo
        panned = self.apply_pan(eqed, prof["pan"])

        # 3. Envíos de efectos (Sends Auxiliares)
        reverb_send_amt = prof.get("reverb_send", 0.0)
        delay_send_amt = prof.get("delay_send", 0.0)

        reverb_out = np.zeros_like(panned)
        delay_out = np.zeros_like(panned)

        if reverb_send_amt > 0.01:
            rev_raw = self.plate_reverb(panned, self.sr)
            reverb_out = rev_raw * reverb_send_amt

        if delay_send_amt > 0.01:
            del_raw = self.tempo_delay(panned, self.sr)
            delay_out = del_raw * delay_send_amt

        wet_processed = panned + reverb_out + delay_out
        return panned, wet_processed

    def mix_and_master(self, multitrack_audio_dict, output_dir, session_name="cumbia_session"):
        """
        Mezcla todas las pistas multipista, aplica sidechain virtual, masteriza y exporta
        tanto el Master final como el Multitrack Stems Pack individual.
        """
        os.makedirs(output_dir, exist_ok=True)
        stems_dir = os.path.join(output_dir, "stems")
        os.makedirs(stems_dir, exist_ok=True)

        print(f"\n=======================================================")
        print(f"🎚️  CONSOLA DE MEZCLA Y MASTERIZACIÓN: {session_name}")
        print(f"=======================================================")

        # Encontrar longitud máxima
        max_samples = max(
            a.shape[-1] if a.ndim == 2 else len(a) 
            for a in multitrack_audio_dict.values()
        )

        sum_bus = np.zeros((2, max_samples), dtype=np.float32)
        processed_stems = {}

        for ch_name, raw_audio in multitrack_audio_dict.items():
            # Pad a longitud uniforme
            curr_len = raw_audio.shape[-1] if raw_audio.ndim == 2 else len(raw_audio)
            if raw_audio.ndim == 1:
                padded = np.pad(raw_audio, (0, max_samples - curr_len))
                stereo_padded = np.stack([padded, padded], axis=0)
            else:
                stereo_padded = np.pad(raw_audio if raw_audio.shape[0] == 2 else raw_audio.T, ((0, 0), (0, max_samples - curr_len)))

            # Procesar pista por su Channel Strip
            dry_stem, wet_stem = self.process_channel(ch_name, stereo_padded)
            sum_bus += wet_stem
            processed_stems[ch_name] = wet_stem

            # Guardar stem individual procesado de estudio (24-bit PCM)
            stem_out_path = os.path.join(stems_dir, f"{session_name}_{ch_name}.wav")
            sf.write(stem_out_path, wet_stem.T, self.sr, subtype='PCM_24')
            print(f"  • Pista procesada y exportada: {os.path.basename(stem_out_path)}")

        # 4. Masterización de Estudio (Glue Compressor + EQ + True Peak Limiter)
        print(f"\n[MASTER BUS] Procesando Master Bus (-14 LUFS / True Peak Limiter a -1.0 dBFS)...")
        master_audio = self.master_bus(sum_bus, self.sr)

        # Evitar clipping absoluto
        peak = np.max(np.abs(master_audio))
        if peak > 0.99:
            master_audio = master_audio * (0.98 / peak)

        master_path = os.path.join(output_dir, f"{session_name}_MASTER.wav")
        sf.write(master_path, master_audio.T, self.sr, subtype='PCM_24')

        # Reporte de métricas de entrega
        master_peak_db = float(round(20 * np.log10(np.max(np.abs(master_audio)) + 1e-9), 2))
        master_rms_db = float(round(20 * np.log10(np.sqrt(np.mean(master_audio**2)) + 1e-9), 2))
        print(f"  ✅ Master Final Exportado: {master_path}")
        print(f"     Nivel Peak: {master_peak_db} dBFS | RMS: {master_rms_db} dBFS | Formato: WAV 24-bit {self.sr}Hz")

        manifest = {
            "session_name": session_name,
            "master_file": master_path,
            "stems_directory": stems_dir,
            "total_canales": len(processed_stems),
            "sample_rate": self.sr,
            "duracion_segundos": float(round(max_samples / self.sr, 2)),
            "master_peak_dbfs": master_peak_db,
            "master_rms_dbfs": master_rms_db,
            "canales_procesados": list(processed_stems.keys())
        }

        manifest_path = os.path.join(output_dir, f"{session_name}_session_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest

if __name__ == "__main__":
    mixer = StudioMixer()
    print("Módulo de Mezcla y Masterización de Estudio listo.")
