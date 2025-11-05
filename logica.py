import torch
import torchhd

dim = 10000

# ---- BASE HYPERVECTORS ----
EsMadreDe, EsPadreDe, EsMujer, EsHombre = torchhd.random(4, dim)
Ana, Carlos, Luis = torchhd.random(3, dim)

IMPLICA, NEG, FORALL, VAR = torchhd.random(4, dim)

# ---- HECHOS ----
hecho_madre = torchhd.bind(EsMadreDe, torchhd.bind(Ana, Luis))   # Madre(Ana, Luis)
hecho_padre = torchhd.bind(EsPadreDe, torchhd.bind(Carlos, Luis))# Padre(Carlos, Luis)

memoria_hechos = torchhd.bundle(hecho_madre, hecho_padre)

# ---- AXIOMAS ----
# ∀x∀y (EsMadreDe(x,y) → EsMujer(x))
axioma_madre_mujer = torchhd.bind(FORALL,torchhd.bind(EsMadreDe, torchhd.bind(IMPLICA, EsMujer)))

# ∀x∀y (EsPadreDe(x,y) → EsHombre(x))
axioma_padre_hombre = torchhd.bind(FORALL,torchhd.bind(EsPadreDe, torchhd.bind(IMPLICA, EsHombre)))

memoria_axiomas = torchhd.bundle(axioma_madre_mujer, axioma_padre_hombre)

# ---- MEMORIA GLOBAL ----
memoria = torchhd.bundle(memoria_hechos, memoria_axiomas)

# ---- CONSULTA ----
# Pregunta: ¿Podemos inferir EsMujer(Ana)?
query = torchhd.bind(EsMujer, Ana)

# ---- INFERENCIA (muy simplificada) ----
# Derivar: HechoMadre ⊗ AxiomaMamásSonMujeres
inferencia = torchhd.bind(hecho_madre, axioma_madre_mujer)

# Comparamos la inferencia con "EsMujer ⊗ Ana"
similaridad = torchhd.cosine_similarity(inferencia, query)

print(f"Similitud (¿Ana es mujer?): {similaridad}")

"""
EsMadreDe * Ana * Luis * FORALL * EsMadreDe * IMPLICA * EsMujer
EsMujer * Ana
"""