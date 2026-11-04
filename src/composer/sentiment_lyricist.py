import re
import json

class SentimentLyricistEngine:
    """
    Motor de Modelado de Sentimiento y Creación Lírica para Cumbia.
    
    Estructura la canción bajo una arquitectura narrativa profesional:
      1. Idea Principal (Tesis Emocional Central / Logline)
      2. Desglose en Ideas Secundarias concretas (Conflicto, Atmósfera, Catarsis en Estribillo, Resolución)
      3. Métrica Poética Octosilábica con rima en español
      4. Marcas explícitas de prosodia, respiración y contestación instrumental
    """

    NARRATIVE_TEMPLATES = {
        "desamor_desgarrador": {
            "idea_principal": "El tormento de ver a quien amas con otra persona en el mismo baile donde se juraron amor eterno.",
            "ideas_secundarias": [
                "1. Atmósfera nocturna: El humo del boliche, el vaso en la mano y la mirada que encuentra lo inesperado.",
                "2. El shock del encuentro: Ver sus manos entrelazadas y la sonrisa que antes era tuya.",
                "3. Clímax emocional (Estribillo): El grito desesperado al corazón pidiendo olvidar lo inolvidable.",
                "4. Catarsis instrumental: El acordeón llora las palabras que ya no se pueden decir."
            ],
            "valence": -0.85,
            "arousal": 0.65,
            "tension": 0.90,
            "sections": {
                "intro": "(El acordeón arranca en notas menores, anunciando la tristeza de la noche)",
                "verso_1": [
                    "Esta noche entré al boliche sin pensar",
                    "Con las ganas de apagar este dolor",
                    "(Silencio de respiración / El acordeón gime con sentimiento)",
                    "Pero al fondo te vi sonreír feliz",
                    "Abrazada con el que era mi mejor amigo"
                ],
                "estribillo": [
                    "¡Olvídala, me grita el corazón!",
                    "¡Olvídala, que ya no tiene dueño tu perdón!",
                    "Aunque me queme el pecho esta traición",
                    "Seguiré bailando con mi desilusión"
                ],
                "verso_2": [
                    "La botella ya se empieza a terminar",
                    "Y el dolor se me clava más y más",
                    "(Silencio de respiración / Contestación del solista)",
                    "Mírame a los ojos si te atreves ya",
                    "Que esta cumbia triste nunca morirá"
                ],
                "punteo_solo": "(Solo desgarrador de acordeón santafesino; el bajo caminante marca la marcha fúnebre)",
                "outro": "Olvídala... que ya nunca volverá. Suena Los Palmeras en el corazón."
            }
        },
        "fiesta_baile": {
            "idea_principal": "La fiesta colectiva de barrio como vía de escape y celebración de la vida ante las penas.",
            "ideas_secundarias": [
                "1. La previa y la llegada: Los amigos reuniéndose, las heladeras con hielo y el sonido de las pailas.",
                "2. La pista que explota: El güiro marcando el paso que nadie puede resistir.",
                "3. Clímax (Estribillo): La arenga compartida de manos arriba olvidando todos los problemas.",
                "4. Celebración pura: El solo de acordeón liderando el baile hasta el amanecer."
            ],
            "valence": 0.90,
            "arousal": 0.95,
            "tension": 0.20,
            "sections": {
                "intro": "(Güiro continuo bien raspado y timbales abriendo la fiesta santafesina)",
                "verso_1": [
                    "Ya se armó la bailanta en el rincón",
                    "Traigan vino y cerveza sin parar",
                    "(Silencio de respiración / El acordeón repica picante)",
                    "Que esta noche se festeja con pasión",
                    "Y las penas las dejamos de lado al entrar"
                ],
                "estribillo": [
                    "¡Baila mi cumbia, gozala de verdad!",
                    "Con las manos arriba hasta el amanecer",
                    "Que retumbe el bombo en toda la ciudad",
                    "Esta fiesta linda nunca va a caer"
                ],
                "verso_2": [
                    "Mueve la cintura al compás del timbal",
                    "Que la cumbia santafesina no tiene igual",
                    "(Silencio de respiración / Contestación rápida del fuelle)",
                    "Grita fuerte que se sienta en el tablón",
                    "Que esta noche somos pura diversión"
                ],
                "punteo_solo": "(Tramo bailable vertiginoso; repiques de timbal y campana a pleno)",
                "outro": "¡Y suena bien arriba! Cumbia santafesina para todo el país."
            }
        },
        "nostalgia_desamor": {
            "idea_principal": "La despedida silenciosa y el recuerdo imborrable de un amor que partió en la lejanía.",
            "ideas_secundarias": [
                "1. La partida: La última tarde de lluvia mirando por la ventana.",
                "2. La ausencia: El silencio en la habitación y las cartas guardadas.",
                "3. Clímax (Estribillo): La resignación melancólica de aceptar que no volverá jamás.",
                "4. Catarsis: La guitarra con delay llorando notas sostenidas."
            ],
            "valence": -0.60,
            "arousal": 0.40,
            "tension": 0.70,
            "sections": {
                "intro": "(Guitarra eléctrica con reverb y delay nostálgico estilo Los Del Bohío)",
                "verso_1": [
                    "Se fue la tarde y la lluvia comenzó",
                    "Y tu recuerdo me vino a buscar",
                    "(Silencio de respiración / La guitarra contesta con eco)",
                    "Una foto vieja sobre el aparador",
                    "Y tu promesa que el viento se llevó"
                ],
                "estribillo": [
                    "Adiós mi amor, adiós para siempre",
                    "Te llevaré grabado en mi mente",
                    "Aunque me duela saberte tan lejos",
                    "En esta cumbia te dejo mis rezos"
                ],
                "verso_2": [
                    "Camino solo la calle desierta",
                    "Sabiendo que no abrirás esa puerta",
                    "(Silencio de respiración / Solo de guitarra)",
                    "Quédate en paz allá donde estés",
                    "Que yo te amaré por última vez"
                ],
                "punteo_solo": "(Solo virtuoso y lento de guitarra con trémolo y feeling santafesino)",
                "outro": "Adiós lady... adiós amor. Queda la noche en silencio."
            }
        }
    }

    def generate_emotional_concept(self, mood="desamor_desgarrador", user_prompt=None):
        """
        Genera el concepto integral de la canción con idea principal,
        desglose de ideas secundarias y letra poética estructurada.
        """
        selected_key = "desamor_desgarrador"
        if mood:
            m = mood.lower()
            if "fiesta" in m or "baile" in m or "alegre" in m:
                selected_key = "fiesta_baile"
            elif "nostalgia" in m or "guitarra" in m or "bohio" in m:
                selected_key = "nostalgia_desamor"

        template = self.NARRATIVE_TEMPLATES[selected_key]
        
        # Si el usuario mandó un prompt temático específico, adaptamos la idea principal
        idea_principal = template["idea_principal"]
        if user_prompt and len(user_prompt) > 5:
            idea_principal = f"Inspirada en '{user_prompt}': {template['idea_principal']}"

        concept = {
            "mood": selected_key,
            "idea_principal": idea_principal,
            "ideas_secundarias": template["ideas_secundarias"],
            "vector_emocional": {
                "valence": template["valence"],
                "arousal": template["arousal"],
                "tension": template["tension"]
            },
            "sections": template["sections"]
        }

        print("\n" + "=" * 65)
        print("📖 CONCEPTO Y NARRATIVA EMOCIONAL DE LA CANCIÓN")
        print("=" * 65)
        print(f"  💡 IDEA PRINCIPAL: {concept['idea_principal']}")
        print(f"  📌 IDEAS SECUNDARIAS:")
        for isec in concept["ideas_secundarias"]:
            print(f"     • {isec}")
        print(f"  📊 VECTOR EMOCIONAL: Valence={concept['vector_emocional']['valence']}, Arousal={concept['vector_emocional']['arousal']}, Tensión={concept['vector_emocional']['tension']}")

        return concept

    def get_full_lyrics_text(self, concept):
        sec = concept["sections"]
        text_lines = []
        text_lines.append("[Intro]")
        text_lines.append(sec["intro"])
        text_lines.append("\n[Verso 1]")
        text_lines.extend(sec["verso_1"])
        text_lines.append("\n[Estribillo]")
        text_lines.extend(sec["estribillo"])
        text_lines.append("\n[Verso 2]")
        text_lines.extend(sec["verso_2"])
        text_lines.append("\n[Tramo Bailable - Punteo]")
        text_lines.append(sec["punteo_solo"])
        text_lines.append("\n[Outro]")
        text_lines.append(sec["outro"])
        return "\n".join(text_lines)

if __name__ == "__main__":
    engine = SentimentLyricistEngine()
    c = engine.generate_emotional_concept("desamor_desgarrador")
    print("\nTexto completo de letra generado:")
    print(engine.get_full_lyrics_text(c))
