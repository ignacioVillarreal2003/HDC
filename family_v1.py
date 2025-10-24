import torch
import torchhd

dimension = 10000

Father, Mother = torchhd.random(2, dimension)
Ignacio, Priscilla, Damian, Daniela = torchhd.random(4, dimension)

ignacio_record = torchhd.bind(
    Ignacio,
    torchhd.bundle(
        torchhd.bind(Father, Damian),
        torchhd.bind(Mother, Daniela)
    )
)

priscilla_record = torchhd.bind(
    Priscilla,
    torchhd.bundle(
        torchhd.bind(Father, Damian),
        torchhd.bind(Mother, Daniela)
    )
)

memory = torchhd.bundle(ignacio_record, priscilla_record)

ignacio_parents = torchhd.bind(memory, Ignacio)
ignacio_father = torchhd.bind(ignacio_parents, Father)
ignacio_mother = torchhd.bind(ignacio_parents, Mother)

people = torch.stack([Damian, Daniela, Ignacio, Priscilla])
people_labels = ["Damian", "Daniela", "Ignacio", "Priscilla"]

father_sims = torchhd.cosine_similarity(ignacio_father, people)
mother_sims = torchhd.cosine_similarity(ignacio_mother, people)

print("\nPadre de Ignacio:")
for label, sim in zip(people_labels, father_sims.tolist()):
    print(f"{label:12s}: {sim:.4f}")

print("\nMadre de Ignacio:")
for label, sim in zip(people_labels, mother_sims.tolist()):
    print(f"{label:12s}: {sim:.4f}")

same_parents_pattern = torchhd.bundle(
    torchhd.bind(Father, Damian),
    torchhd.bind(Mother, Daniela)
)

siblings_result = torchhd.bind(memory, same_parents_pattern)

siblings_sims = torchhd.cosine_similarity(siblings_result, people)

print("\nPersonas con los mismos padres:")
for label, sim in zip(people_labels, siblings_sims.tolist()):
    print(f"{label:12s}: {sim:.4f}")