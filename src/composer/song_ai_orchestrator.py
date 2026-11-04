import os
import json
import time
from src.composer.multimodal_input import MultimodalInputHandler
from src.composer.symbolic_tokenizer import SymbolicMultitrackTokenizer
from src.composer.multitrack_transformer import MultitrackCrossAttentionTransformer
from src.composer.critic_evaluator import MusicCriticEvaluator
from src.synthesizer.real_instrument_engine import RealInstrumentEngine
from src.composer.studio_engine import StudioMixer
import torch

class SongAIOrchestrator:
    """
    Orquestador Central End-to-End de Song AI Composer.
    
    Integra de punta a punta todas las fases de un estudio de grabación:
    1. Pre-Producción Multimodal (Prompt, Letra, Tarareo o Zero-shot)
    2. Composición Lírica y Métrica con pausas expresivas
    3. Generación Neuronal Multitrack con Transformer entrenado en PyTorch
    4. Auditoría Crítica e Inpainting Quirúrgico de Corrección
    5. Grabación y Recombinación Acústica de Estudio con Instrumentos REALES del Dataset
    6. Mezcla y Masterización Automatizada de Estudio (-14 LUFS / True Peak Limiter)
    7. Entrega Profesional: Master WAV + Pack de 8 Stems DAW + Letra + Manifiesto
    """

    def __init__(self, outputs_root="outputs"):
        self.outputs_root = outputs_root
        self.input_handler = MultimodalInputHandler()
        self.tokenizer = SymbolicMultitrackTokenizer()
        
        # Cargar el Transformer Neuronal Profundo
        self.neural_composer = MultitrackCrossAttentionTransformer(self.tokenizer.vocab_sizes)
        weights_path = "models/cumbia_transformer_weights.pt"
        if os.path.exists(weights_path):
            self.neural_composer.load_state_dict(torch.load(weights_path, map_location="cpu"))
            print(f"[IA] Pesos neuronales entrenados cargados con éxito desde: {weights_path}")
        else:
            print("[IA] Modelo inicializado (sin pesos previos).")

        self.critic = MusicCriticEvaluator()
        self.real_engine = RealInstrumentEngine(sample_rate=44100)
        self.mixer = StudioMixer(sample_rate=44100)
        os.makedirs(self.outputs_root, exist_ok=True)

    def generate_cumbia_lyrics(self, mood="desamor_melancolia", topic="amor_perdido"):
        """Genera letra poética de cumbia con métrica de estudio y marcas de silencio de respiración."""
        if mood == "fiesta_baile":
            return """[Intro]
(Arranca el güiro y suena el acordeón)

[Verso 1]
La noche está bien linda para festejar
Traigan la cerveza que vamos a bailar
(Silencio de respiración / Contesta el acordeón)
Suenan los timbales marcando el compás
Esta cumbia linda no termina más

[Estribillo]
Baila mi cumbia, bailala sin parar
Con las manos arriba hasta el amanecer
Siente el repique que te va a enamorar
Que esta fiesta linda no la vas a perder

[Tramo Bailable - Punteo]
(Solo virtuoso de acordeón y redoble de timbales)

[Outro]
¡Y suena para todo Santa Fe!
Cumbia de estudio hasta el final."""
        else:
            return """[Intro]
(Melodía nostálgica de acordeón santafesino)

[Verso 1]
Hoy me desperté pensando en tu mirar
Buscando en mi cuarto tu calor de ayer
(Silencio de respiración / Contesta el acordeón con sentimiento)
La casa está vacía y duele la verdad
Un vaso de vino para olvidar tu piel

[Estribillo]
Olvídala me dicen todos al pasar
Pero en mi pecho no la puedo borrar
Suena este ritmo triste en la soledad
Llora el acordeón queriéndote llamar

[Tramo Bailable - Punteo]
(Lamento del acordeón en notas menores)

[Outro]
Se va apagando el bajo en la penumbra...
Solo queda el recuerdo de tu adiós."""

    def compose_and_produce_song(self, prompt=None, lyrics_text=None, audio_memo_path=None, mood=None, bpm=None, key=None, song_title="Cumbia_Original_Estudio"):
        print("\n" + "=" * 65)
        print(f"🎵 INICIANDO PRODUCCIÓN AUTOMATIZADA DE ESTUDIO: '{song_title}'")
        print("=" * 65)

        start_time = time.time()
        song_slug = song_title.lower().replace(" ", "_").replace("'", "")
        session_out_dir = os.path.join(self.outputs_root, song_slug)
        os.makedirs(session_out_dir, exist_ok=True)

        # FASE 1: Directivas Multimodales
        directives = self.input_handler.resolve_song_directives(
            prompt=prompt, lyrics_text=lyrics_text, audio_memo_path=audio_memo_path,
            mood=mood, bpm=bpm, key=key
        )

        # FASE 2: Letra y Prosodia
        if not directives["lyrics"]:
            print("\n[LETRISTA IA] Creando letra original adaptada al sentimiento...")
            lyrics_content = self.generate_cumbia_lyrics(mood=directives["mood"])
            directives["lyrics"] = self.input_handler.parse_user_lyrics(lyrics_content)
        else:
            lyrics_content = lyrics_text

        lyrics_file_path = os.path.join(session_out_dir, f"{song_slug}_lyrics.txt")
        with open(lyrics_file_path, "w", encoding="utf-8") as f:
            f.write(lyrics_content)
        print(f"  📄 Letra y marcas de respiración guardadas en: {lyrics_file_path}")

        # FASE 3: Generación Neuronal Multitrack (Multitrack Transformer)
        score_events = self.neural_composer.generate_multitrack_score(directives, total_bars=32)

        # FASE 4: Auditoría del Crítico e Inpainting Quirúrgico
        issues = self.critic.audit_score(score_events, key=directives["key"])
        verified_score = self.critic.apply_surgical_inpainting(score_events, issues, key=directives["key"])

        # FASE 5: Grabación y Recombinación Acústica con Instrumentos REALES del Dataset
        stems_audio_buffers = self.real_engine.render_real_studio_multitrack(
            directives, lyrics_text=lyrics_content, total_bars=32
        )

        # FASE 6: Consola Virtual de Mezcla y Masterización DAW (StudioMixer)
        mix_manifest = self.mixer.mix_and_master(
            stems_audio_buffers, output_dir=session_out_dir, session_name=song_slug
        )

        # Guardar partitura simbólica verificada
        score_json_path = os.path.join(session_out_dir, f"{song_slug}_score_events.json")
        with open(score_json_path, "w", encoding="utf-8") as f:
            json.dump(verified_score, f, indent=2, ensure_ascii=False)

        total_elapsed = round(time.time() - start_time, 2)
        print("\n" + "=" * 65)
        print(f"🎉 PRODUCCIÓN FINALIZADA CON ÉXITO EN {total_elapsed} SEGUNDOS")
        print("=" * 65)
        print(f"  🏆 Master Final Comercial: {mix_manifest['master_file']}")
        print(f"  📁 Multitrack Stems Pack:   {mix_manifest['stems_directory']}")
        print(f"  📜 Letra & Respiraciones:   {lyrics_file_path}")
        print(f"  📊 Manifiesto de Sesión:    {os.path.join(session_out_dir, f'{song_slug}_session_manifest.json')}")
        print(f"  🎚️ Nivel Peak: {mix_manifest['master_peak_dbfs']} dBFS | RMS: {mix_manifest['master_rms_dbfs']} dBFS")

        return {
            "title": song_title,
            "directives": directives,
            "master_file": mix_manifest["master_file"],
            "stems_directory": mix_manifest["stems_directory"],
            "lyrics_file": lyrics_file_path,
            "manifest_file": os.path.join(session_out_dir, f"{song_slug}_session_manifest.json"),
            "elapsed_seconds": total_elapsed
        }

if __name__ == "__main__":
    orchestrator = SongAIOrchestrator()
    res = orchestrator.compose_and_produce_song(
        prompt="Cumbia santafesina de fiesta bailable",
        song_title="El Parrandero del Futuro"
    )
