import torch
import torchhd

dimenciones = 10000


# VECTORES BASE
futbol, tenis, natacion = torchhd.random(3, dimenciones)


# VECTORES DE ATRIBUTOS
en_equipo, en_solitario = torchhd.random(2, dimenciones)
en_cesped, en_cemento, en_piscina = torchhd.random(3, dimenciones)
usa_pelota, usa_raqueta, no_usa_nada = torchhd.random(3, dimenciones)
tiene_contacto, no_tiene_contacto = torchhd.random(2, dimenciones)
duracion_larga, duracion_media, duracion_corta = torchhd.random(3, dimenciones)
habilidad_de_coordinacion, habilidad_de_velocidad, habilidad_de_tecnica, habilidad_de_resistencia = torchhd.random(4, dimenciones)


# CREACIÓN DE CONCEPTOS

# FÚTBOL
futbol_atributos = torchhd.bundle(
    torchhd.bind(futbol, en_equipo),
    torchhd.bind(futbol, en_cesped)
).bundle(
    torchhd.bind(futbol, usa_pelota)
).bundle(
    torchhd.bind(futbol, tiene_contacto)
).bundle(
    torchhd.bind(futbol, duracion_larga)
).bundle(
    torchhd.bind(futbol, habilidad_de_coordinacion)
).bundle(
    torchhd.bind(futbol, habilidad_de_velocidad)
)

# TENIS
tenis_atributos = torchhd.bundle(
    torchhd.bind(tenis, en_solitario),
    torchhd.bind(tenis, en_cemento)
).bundle(
    torchhd.bind(tenis, usa_raqueta)
).bundle(
    torchhd.bind(tenis, no_tiene_contacto)
).bundle(
    torchhd.bind(tenis, duracion_media)
).bundle(
    torchhd.bind(tenis, habilidad_de_tecnica)
)

# NATACIÓN
natacion_atributos = (torchhd.bundle(
    torchhd.bind(natacion, en_solitario),
    torchhd.bind(natacion, en_piscina)
).bundle(
    torchhd.bind(natacion, no_usa_nada)
).bundle(
    torchhd.bind(natacion, no_tiene_contacto)
).bundle(
    torchhd.bind(natacion, duracion_corta)
).bundle(
    torchhd.bind(natacion, habilidad_de_velocidad)
))

# MEMORIA
memoria = torchhd.bundle(futbol_atributos, tenis_atributos).bundle(
    natacion_atributos)


# CONSULTA 1: ¿Deporte individual sin contacto?

print("¿Deporte individual sin contacto?")

consulta_1 = torchhd.bundle(en_solitario, no_tiene_contacto)
resultado_1 = torchhd.bind(memoria, consulta_1)

valores_deportes = torch.stack([futbol, tenis, natacion])
etiquetas_deportes = ["futbol", "tenis", "natacion"]

similardades_1 = torchhd.cosine_similarity(resultado_1, valores_deportes)

for etiqueta, sim in zip(etiquetas_deportes, similardades_1.tolist()):
    print(f"{etiqueta:20s}: {sim:.4f}")

# CONSULTA 2: ¿Deportes que requieren resistencia?

print("\n¿Deportes que requieren resistencia?")

resultado_2 = torchhd.bind(memoria, habilidad_de_resistencia)

similardades_2 = torchhd.cosine_similarity(resultado_2, valores_deportes)

for etiqueta, sim in zip(etiquetas_deportes, similardades_2.tolist()):
    print(f"{etiqueta:20s}: {sim:.4f}")

# CONSULTA 3: ¿Deportes con coordinación Y velocidad?

print("\n¿Deportes con coordinación Y velocidad?")

consulta_3 = torchhd.bundle(habilidad_de_coordinacion, habilidad_de_velocidad)
resultado_3 = torchhd.bind(memoria, consulta_3)

similardades_3 = torchhd.cosine_similarity(resultado_3, valores_deportes)

for etiqueta, sim in zip(etiquetas_deportes, similardades_3.tolist()):
    print(f"{etiqueta:20s}: {sim:.4f}")

# CONSULTA 4: ¿Deportes con coordinación O velocidad?

print("\n¿Deportes con coordinación O velocidad?")

resultado_4_a = torchhd.bind(memoria, habilidad_de_coordinacion)
resultado_4_b = torchhd.bind(memoria, habilidad_de_velocidad)

similardades_4_a = torchhd.cosine_similarity(resultado_4_a, valores_deportes)
similardades_4_b = torchhd.cosine_similarity(resultado_4_b, valores_deportes)

for etiqueta, sim in zip(etiquetas_deportes, similardades_4_a.tolist()):
    print(f"{etiqueta:20s}: {sim:.4f}")

for etiqueta, sim in zip(etiquetas_deportes, similardades_4_b.tolist()):
    print(f"{etiqueta:20s}: {sim:.4f}")

# CONSULTA 5: ¿De qué trata el tenis?

print("\n¿De qué trata el tenis?")

resultado_5 = torchhd.bind(memoria, tenis)

valores_atributos = torch.stack([
    en_equipo, en_solitario, en_cesped, en_cemento, en_piscina,
    usa_pelota, usa_raqueta, no_usa_nada,
    tiene_contacto, no_tiene_contacto,
    duracion_larga, duracion_media, duracion_corta,
    habilidad_de_coordinacion, habilidad_de_velocidad, habilidad_de_tecnica, habilidad_de_resistencia
])
etiquetas_atributos = ["en_equipo", "en_solitario",
    "en_cesped", "en_cemento", "en_piscina",
    "usa_pelota", "usa_raqueta", "no_usa_nada",
    "tiene_contacto", "no_tiene_contacto",
    "duracion_larga", "duracion_media", "duracion_corta",
    "habilidad_de_coordinacion", "habilidad_de_velocidad", "habilidad_de_tecnica", "habilidad_de_resistencia"]

similardades_5 = torchhd.cosine_similarity(resultado_5, valores_atributos)

for etiqueta, sim in zip(etiquetas_atributos, similardades_5.tolist()):
    print(f"{etiqueta:20s}: {sim:.4f}")