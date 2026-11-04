import json
import numpy as np

class GlobalSongPlanner:
    """
    Planificador de Coherencia Global y Armonía Matemática (Director de Orquesta IA).
    
    Resuelve el problema del 'cocoliche' asegurando que la canción sea una unidad coherente:
      1. Genera un Motivo Melódico Central (Tema A / Hook) que da identidad a toda la obra.
      2. Aplica la Forma Canónica Completa:
         Intro (Tema A) -> Verso 1 -> Pre-Coro -> Estribillo -> VUELTA A INTRO (Tema A)
         -> Verso 2 -> Estribillo -> Solo Virtuoso (Variación Tema A) -> Outro (Tema A)
      3. Rige la Armonía Matemática: Cadencias funcionales (i -> iv -> V7 -> i),
         conducción de voces sin choques y resolución armónica estricta.
      4. Controla la dinámica de conjunto: Bombo y timbales entran y salen en bloque
         según la tensión emocional de cada sección.
    """

    NOTES_PITCH = {
        "C": 60, "C#": 61, "D": 62, "D#": 63, "E": 64, "F": 65,
        "F#": 66, "G": 67, "G#": 68, "A": 69, "A#": 70, "B": 71
    }

    def __init__(self):
        pass

    def generate_thematic_motif_hook(self, root_pitch=62, is_minor=True):
        """
        Diseña matemáticamente el 'Tema A' (Motivo Principal de 4 compases):
        Es la melodía inolvidable que suena en la Intro, vuelve en la Re-Intro,
        se transforma en el Solo y cierra la canción.
        """
        # Escala menor: [0, 2, 3, 5, 7, 8, 11] (armónica con sensible resolutiva)
        third = 3 if is_minor else 4
        fifth = 7
        sixth = 8 if is_minor else 9
        leading = 11 # Sensible que resuelve al tono

        # Frase temática cumbiera en semicorcheas (Pregunta y Respuesta interna de 4 compases)
        # Compás 1: Afirmación del tono (Tónica -> Tercera -> Quinta)
        c1 = [root_pitch, root_pitch + third, root_pitch + fifth, root_pitch + fifth, root_pitch + third]
        # Compás 2: Movimiento a la subdominante (iv)
        c2 = [root_pitch + 5, root_pitch + sixth, root_pitch + 5, root_pitch + third]
        # Compás 3: Tensión en la dominante (V7 con sensible)
        c3 = [root_pitch + fifth, root_pitch + leading, root_pitch + fifth, root_pitch + third]
        # Compás 4: Resolución contundente a la tónica
        c4 = [root_pitch + 2, root_pitch, root_pitch, root_pitch]

        return [c1, c2, c3, c4]

    def build_canonical_song_architecture(self, key="Dm", bpm=100.0, mood="desamor_desgarrador"):
        """
        Genera el mapa estructural completo de una canción de cumbia coherente (56 compases).
        """
        is_minor = key.endswith("m")
        root_name = key.replace("m", "")
        root_pitch = self.NOTES_PITCH.get(root_name, 62)

        # Progresión armónica funcional estricta:
        # i (Tónica) -> iv (Subdominante) -> V7 (Dominante con tensión) -> i (Resolución)
        chords_verse = [root_pitch, root_pitch + 5, root_pitch + 7, root_pitch]
        chords_chorus = [root_pitch + (8 if is_minor else 5), root_pitch + 5, root_pitch + 7, root_pitch]

        # Motivo Temático Central
        motif_A = self.generate_thematic_motif_hook(root_pitch, is_minor)

        # Forma Canónica Completa: 56 compases
        # 1. Intro (4 compases): Tema A completo
        # 2. Verso 1 (8 compases): Intimidad lírica + contestaciones
        # 3. Pre-coro (4 compases): Tensión hacia el coro
        # 4. Estribillo (8 compases): Clímax de energía con campana
        # 5. RE-INTRO (4 compases): VUELTA AL TEMA A (Anclaje de identidad)
        # 6. Verso 2 (8 compases): Segunda estrofa
        # 7. Estribillo (8 compases): Segundo clímax
        # 8. Solo / Tramo Bailable (8 compases): Variación virtuosa del Tema A
        # 9. Outro (4 compases): Cierre con Tema A y resolución final

        sections_blueprint = [
            {"id": "INTRO", "bars": 4, "role": "exposicion_tema_A", "bombo_active": True, "lead_theme": "tema_A", "vox_active": False},
            {"id": "VERSO_1", "bars": 8, "role": "narrativa_intima", "bombo_active": False, "lead_theme": "contestacion", "vox_active": True},
            {"id": "PRE_CORO", "bars": 4, "role": "tension_subdominante", "bombo_active": True, "lead_theme": "tension", "vox_active": True},
            {"id": "ESTRIBILLO", "bars": 8, "role": "climax_catartico", "bombo_active": True, "lead_theme": "armonia_llena", "vox_active": True},
            {"id": "RE_INTRO", "bars": 4, "role": "vuelta_al_tema_A", "bombo_active": True, "lead_theme": "tema_A", "vox_active": False},
            {"id": "VERSO_2", "bars": 8, "role": "desenlace_narrativo", "bombo_active": False, "lead_theme": "contestacion", "vox_active": True},
            {"id": "ESTRIBILLO_2", "bars": 8, "role": "climax_catartico", "bombo_active": True, "lead_theme": "armonia_llena", "vox_active": True},
            {"id": "SOLO_BAILABLE", "bars": 8, "role": "virtuosismo_tema_A", "bombo_active": True, "lead_theme": "variacion_virtuosa", "vox_active": False},
            {"id": "OUTRO", "bars": 4, "role": "resolucion_coda", "bombo_active": True, "lead_theme": "tema_A_final", "vox_active": False}
        ]

        total_bars = sum(s["bars"] for s in sections_blueprint)

        plan = {
            "key": key,
            "root_pitch": root_pitch,
            "is_minor": is_minor,
            "bpm": bpm,
            "mood": mood,
            "total_bars": total_bars,
            "motif_A_notes": motif_A,
            "sections": sections_blueprint,
            "duracion_total_segundos": round((60.0 / bpm) * 4.0 * total_bars, 1)
        }

        print("\n" + "=" * 65)
        print(f"🎼 ARQUITECTURA GLOBAL DE LA CANCIÓN: {key} ({bpm} BPM)")
        print("=" * 65)
        print(f"  • Estructura Canónica: {total_bars} compases ({plan['duracion_total_segundos']} seg)")
        print(f"  • Motivo Central (Tema A): Configurado en Intro, Re-Intro, Solo y Coda.")
        for s in sections_blueprint:
            print(f"    -> [{s['id']:<14}] | {s['bars']:>2} compases | Bombo: {'Activo' if s['bombo_active'] else 'Descanso':<8} | Lead: {s['lead_theme']:<18} | Voz: {'Canta' if s['vox_active'] else 'Silencio'}")

        return plan

if __name__ == "__main__":
    planner = GlobalSongPlanner()
    plan = planner.build_canonical_song_architecture(key="Dm", bpm=102.0)
    print("\nPlanificador global probado con éxito.")
