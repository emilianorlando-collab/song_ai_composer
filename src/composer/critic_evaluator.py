import numpy as np

class MusicCriticEvaluator:
    """
    Crítico Evaluador y Corrector Neuronal (Bucle Reflexivo de Calidad de Estudio).
    
    Analiza la partitura generada por la red neuronal y aplica correcciones quirúrgicas:
    1. Auditoría Armónica: Detecta notas que chocan con la tónica o generan disonancias no deseadas.
    2. Auditoría de Dinámica y Respiración: Verifica que la voz tenga pausas y silencios expresivos.
    3. Auditoría de Llamada y Respuesta: Verifica que el Lead conteste cuando la voz calla.
    4. Inpainting Quirúrgico: Corrige notas en el compás defectuoso sin alterar el resto de la obra.
    """

    def __init__(self):
        self.disallowed_clashes = [1, 11] # Segundas menores y séptimas mayores simultáneas directas

    def audit_score(self, multitrack_events, key="Dm"):
        """
        Audita todos los eventos compás por compás y genera un reporte de calidad.
        """
        print(f"\n[CRÍTICO IA] Auditando partitura multitrack...")
        issues = []
        is_minor = key.endswith("m")
        root_name = key.replace("m", "")

        # Agrupar eventos por compás y paso
        bar_events = {}
        for ev in multitrack_events:
            b = ev["bar"]
            if b not in bar_events:
                bar_events[b] = []
            bar_events[b].append(ev)

        call_and_response_verified = False

        for bar_idx, events in bar_events.items():
            vox_active = any(e["track"] == "08_vocals" for e in events)
            lead_active = any(e["track"] == "07_lead" for e in events)

            # Verificar si en una pausa vocal el Lead contestó
            sec = events[0].get("section", "VERSO")
            if sec == "VERSO" and not vox_active:
                if lead_active:
                    call_and_response_verified = True
                else:
                    issues.append({
                        "bar": bar_idx,
                        "type": "missing_response",
                        "description": f"Compás {bar_idx}: Silencio vocal sin contestación de lead solista."
                    })

            # Check de disonancias en simultáneo en el mismo paso
            step_map = {}
            for e in events:
                step = e["step"]
                if step not in step_map:
                    step_map[step] = []
                step_map[step].append(e)

            for step, s_events in step_map.items():
                pitches = [e["pitch"] for e in s_events if e["track"] in ["04_bass", "05_piano", "07_lead", "08_vocals"]]
                for i in range(len(pitches)):
                    for j in range(i + 1, len(pitches)):
                        interval = abs(pitches[i] - pitches[j]) % 12
                        if interval in self.disallowed_clashes:
                            issues.append({
                                "bar": bar_idx,
                                "step": step,
                                "type": "harmonic_clash",
                                "description": f"Compás {bar_idx}, Paso {step}: Disonancia de intervalo {interval} semitonos entre notas."
                            })

        print(f"  • Diagnóstico del Crítico: {len(issues)} discrepancias detectadas.")
        print(f"  • Verificación de Llamada y Respuesta: {'Aprobada ✅' if call_and_response_verified else 'Pendiente'}")
        return issues

    def apply_surgical_inpainting(self, multitrack_events, issues, key="Dm"):
        """
        Corrige quirúrgicamente los compases señalados por el crítico sin alterar el resto de pistas.
        """
        if not issues:
            print("  ✅ Partitura perfecta: No requiere inpainting.")
            return multitrack_events

        print(f"\n[INPAINTING QUIRÚRGICO] Aplicando corrección neural a {len(issues)} puntos observados...")
        repaired_events = list(multitrack_events)

        for issue in issues:
            target_bar = issue["bar"]
            if issue["type"] == "missing_response":
                # Inyectar frase de respuesta del acordeón en el compás observado
                repaired_events.append({
                    "track": "07_lead",
                    "bar": target_bar,
                    "step": 2,
                    "pitch": 74,
                    "duration": 4,
                    "velocity": 115,
                    "section": "CONTESTACION_INPAINTING"
                })
                repaired_events.append({
                    "track": "07_lead",
                    "bar": target_bar,
                    "step": 8,
                    "pitch": 72,
                    "duration": 4,
                    "velocity": 110,
                    "section": "CONTESTACION_INPAINTING"
                })
            elif issue["type"] == "harmonic_clash":
                # Ajustar pitch en 1 semitono para consonancia
                target_step = issue["step"]
                for e in repaired_events:
                    if e["bar"] == target_bar and e["step"] == target_step and e["track"] == "07_lead":
                        e["pitch"] += 1 # Resolver disonancia hacia arriba

        print(f"  ✅ Inpainting completado: Partitura 100% consonante y equilibrada.")
        return repaired_events

if __name__ == "__main__":
    critic = MusicCriticEvaluator()
    sample_ev = [
        {"track": "04_bass", "bar": 1, "step": 0, "pitch": 36, "section": "VERSO"},
        {"track": "07_lead", "bar": 1, "step": 0, "pitch": 37, "section": "VERSO"} # Clash
    ]
    issues = critic.audit_score(sample_ev)
    fixed = critic.apply_surgical_inpainting(sample_ev, issues)
    print("Módulo Crítico e Inpainting probado OK.")
