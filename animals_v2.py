import torch
import torchhd


# ============================================================
# CONFIGURACIÓN BASE
# ============================================================

dimension = 10000

# Roles
species, habitat, diet, movement = torchhd.random(4, dimension)

# Valores
carnivore, herbivore, omnivore = torchhd.random(3, dimension)
savanna, ocean, mountains, farm, pole, forest = torchhd.random(6, dimension)
terrestrial, aquatic, aerial = torchhd.random(3, dimension)
lion, turtle, eagle, elephant, cow, penguin, monkey = torchhd.random(7, dimension)


# ============================================================
# DEFINICIÓN DE ANIMALES Y SUS ATRIBUTOS
# ============================================================

# León
lion_attributes = torchhd.bundle(torchhd.bind(habitat, savanna),torchhd.bind(diet, carnivore)).bundle(torchhd.bind(movement, terrestrial))
lion_record = torchhd.bind(torchhd.bind(species, lion), lion_attributes)

# Tortuga
turtle_attributes = torchhd.bundle(torchhd.bind(habitat, ocean),torchhd.bind(diet, herbivore)).bundle(torchhd.bind(movement, aquatic))
turtle_record = torchhd.bind(torchhd.bind(species, turtle), turtle_attributes)

# Águila
eagle_attributes = torchhd.bundle(torchhd.bind(habitat, mountains),torchhd.bind(diet, carnivore)).bundle(torchhd.bind(movement, aerial))
eagle_record = torchhd.bind(torchhd.bind(species, eagle), eagle_attributes)

# Elefante
elephant_attributes = torchhd.bundle(torchhd.bind(habitat, savanna),torchhd.bind(diet, herbivore)).bundle(torchhd.bind(movement, terrestrial))
elephant_record = torchhd.bind(torchhd.bind(species, elephant), elephant_attributes)

# Vaca
cow_attributes = torchhd.bundle(torchhd.bind(habitat, farm),torchhd.bind(diet, herbivore)).bundle(torchhd.bind(movement, terrestrial))
cow_record = torchhd.bind(torchhd.bind(species, cow), cow_attributes)

# Pingüino
penguin_attributes = torchhd.bundle(torchhd.bind(habitat, pole),torchhd.bind(diet, carnivore)).bundle(torchhd.bind(movement, aquatic))
penguin_record = torchhd.bind(torchhd.bind(species, penguin), penguin_attributes)

# Mono
monkey_attributes = torchhd.bundle(torchhd.bind(habitat, forest), torchhd.bind(diet, omnivore)).bundle(torchhd.bind(movement, terrestrial))
monkey_record = torchhd.bind(torchhd.bind(species, monkey), monkey_attributes)


# ============================================================
# MEMORIA HD — Contiene todos los animales
# ============================================================

memory_hd = torchhd.bundle(lion_record,turtle_record).bundle(eagle_record).bundle(elephant_record).bundle(cow_record).bundle(penguin_record).bundle(monkey_record)


# ============================================================
# FUNCIÓN DE CONSULTA GENERAL
# ============================================================

def consultar(rol, valor):
    query_key = torchhd.bind(rol, valor)
    attrs = torchhd.bind(memory_hd, query_key)
    return attrs


def decodificar(resultado, valores, etiquetas):
    similarities = torchhd.cosine_similarity(resultado, valores)
    for label, sim in zip(etiquetas, similarities.tolist()):
        print(f"{label:20s}: {sim:.4f}")
    best_match_idx = torch.argmax(similarities).item()
    return etiquetas[best_match_idx]


# ============================================================
# CONSULTAS DE EJEMPLO
# ============================================================

print("¿Cuál es la dieta de la tortuga?")
turtle_attrs = consultar(species, turtle)
result = torchhd.bind(turtle_attrs, diet)
valores = torch.stack([carnivore, herbivore, omnivore])
etiquetas = ["carnivore", "herbivore", "omnivore"]
print("💡 Respuesta:", decodificar(result, valores, etiquetas))

print("¿Cuál es el hábitat del pingüino?")
penguin_attrs = consultar(species, penguin)
result = torchhd.bind(penguin_attrs, habitat)
valores = torch.stack([savanna, ocean, farm, pole, forest, mountains])
etiquetas = ["savanna", "ocean", "farm", "pole", "forest", "mountains"]
print("💡 Respuesta:", decodificar(result, valores, etiquetas))

print("¿Cómo se mueve el elefante?")
elephant_attrs = consultar(species, elephant)
result = torchhd.bind(elephant_attrs, movement)
valores = torch.stack([terrestrial, aquatic, aerial])
etiquetas = ["terrestrial", "aquatic", "aerial"]
print("💡 Respuesta:", decodificar(result, valores, etiquetas))