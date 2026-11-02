# Song AI Composer (Laboratorio de Composición Musical con IA)

Un sistema de inteligencia artificial modular y jerárquico diseñado para componer música auténtica con control compositivo profundo, comenzando con el género de la **cumbia** (santafesina, villera, colombiana y más).

A diferencia de los modelos de audio monolíticos "caja negra", este proyecto separa el proceso en tres niveles fundamentales:
1. **El Oído (Análisis de Audio y Extracción Simbólica):** Separación de stems, detección de pitch a MIDI (incluso desde notas de voz de celular), extracción de tempo, tonalidad y alineación prosódica lírica.
2. **El Cerebro (Motor de Composición y Emoción):** Modelado armónico, patrones rítmicos de síncopa y llamada/respuesta, mapeo emocional de letras a tonalidades y fraseo elástico.
3. **El Estudio (Síntesis y Edición Quirúrgica):** Renderizado de audio en alta fidelidad y modificación selectiva por segmentos (inpainting en compases específicos).

---

## 📁 Estructura del Repositorio y del Disco T7

El repositorio está diseñado para sincronizarse con Git en la nube sin saturar el almacenamiento de audio:

```text
song_ai_composer/
├── data/
│   ├── raw/             # Audios originales (MP3/WAV) - [Local en T7, ignorado por Git]
│   ├── stems/           # Pistas separadas (Demucs) - [Local en T7, ignorado por Git]
│   ├── midi/            # Transcripciones de melodías y acordes (MIDI) - [En Git]
│   ├── lyrics/          # Letras y alineaciones fonéticas (JSON/LRC) - [En Git]
│   ├── processed/       # Matrices y secuencias de entrenamiento - [En Git]
│   └── metadata.csv     # Catálogo con BPM, tono, emoción y estilo de cada canción
├── src/
│   ├── analyzer/        # Extracción de audio, stems, afinación y prosodia
│   ├── composer/        # Generación simbólica, reglas armónicas y métrica
│   ├── synthesizer/     # Renderizado de audio profesional e inpainting
│   └── utils/           # Herramientas auxiliares de audio y conversión
├── outputs/
│   └── audio/           # Renderizados finales generados
├── notebooks/           # Cuadernos de experimentación rápida
├── .gitignore           # Control estricto para no subir gigas de audio binario a GitHub
└── README.md
```

---

## 🎯 Protocolo para el Dataset Inicial

Para evitar que el algoritmo genere una "ensalada" o música acartonada:
* **Cantidad recomendada:** Entre **25 y 30 canciones** seleccionadas con criterio.
* **Composición sugerida:** 
  * 80% (20 a 25 canciones) de un subgénero núcleo (ej. Cumbia Santafesina).
  * 20% (5 a 10 canciones) de contraste (ej. Cumbia Villera o Colombiana).
* **Calidad:** Preferir archivos de audio con producción nítida (192 kbps o superior / WAV).
* **Metadatos:** Cada archivo en `data/raw/` debe tener su correspondiente fila en `data/metadata.csv` especificando subgénero, emoción, BPM y roles instrumentales.
