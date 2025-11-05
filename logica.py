import torch
import torchhd

dim = 10000

EsMadreDe, EsPadreDe, EsMujer, EsHombre = torchhd.random(4, dim)
Ana, Carlos, Luis = torchhd.random(3, dim)
IMPLICA, ARG1, ARG2  = torchhd.random(3, dim)

# EsMadreDe(Ana, Luis)
hecho1 = torchhd.bind(
    EsMadreDe,
    torchhd.bundle(torchhd.bind(ARG1, Ana), torchhd.bind(ARG2, Luis))
)

# EsPadreDe(Carlos, Luis)
hecho2 = torchhd.bind(
    EsPadreDe,
    torchhd.bundle(torchhd.bind(ARG1, Carlos), torchhd.bind(ARG2, Luis))
)

# IMPLICA(EsMadreDe, EsMujer)
regla1 = torchhd.bind(IMPLICA, torchhd.bind(EsMadreDe, EsMujer))

# IMPLICA(EsPadreDe, EsHombre)
regla2 = torchhd.bind(IMPLICA, torchhd.bind(EsPadreDe, EsHombre))

# MEMORIA
memoria = torchhd.bundle(hecho1, hecho2).bundle(regla1).bundle(regla2)

print("¿Ana es mujer?\n")

# memoria * ARG1 * Ana
consulta1 = torchhd.bind(torchhd.bind(memoria, ARG1), Ana)

sim_madre = torchhd.cosine_similarity(consulta1, EsMadreDe)
sim_padre = torchhd.cosine_similarity(consulta1, EsPadreDe)
print(f"EsMadreDe: {sim_madre.item():.4f}")
print(f"EsPadreDe: {sim_padre.item():.4f}")

# memoria * IMPLICA * EsMadreDe
consulta2 = torchhd.bind(torchhd.bind(memoria, IMPLICA), EsMadreDe)

sim_mujer = torchhd.cosine_similarity(consulta2, EsMujer)
sim_hombre = torchhd.cosine_similarity(consulta2, EsHombre)
print(f"\nEsMujer:  {sim_mujer.item():.4f}")
print(f"EsHombre: {sim_hombre.item():.4f}")