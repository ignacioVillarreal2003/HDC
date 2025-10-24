import torch
import torchhd

dimension = 10000

keys = torchhd.random(3, dimension)
country, capital, currency = keys

united_states, mexico = torchhd.random(2, dimension)
washington_dc, mexico_city = torchhd.random(2, dimension)
united_states_dollar, mexican_peso = torchhd.random(2, dimension)

print("=== Entidades (roles) ===")
print("País:", country[:10])
print("Capital:", capital[:10])
print("Moneda:", currency[:10])
print()

print("=== Valores de ejemplo ===")
print("Estados Unidos:", united_states[:10])
print("México:", mexico[:10])
print("Washington D.C.:", washington_dc[:10])
print("Ciudad de México:", mexico_city[:10])
print("Dólar estadounidense:", united_states_dollar[:10])
print("Peso mexicano:", mexican_peso[:10])
print()

# Agrupar valores de cada país
united_states_values = torch.stack([united_states, washington_dc, united_states_dollar])
mexico_values = torch.stack([mexico, mexico_city, mexican_peso])

# Codificar (binding + bundling)
us = torchhd.hash_table(keys, united_states_values)
mx = torchhd.hash_table(keys, mexico_values)

print("=== Hipervectores compuestos ===")
print("Estados Unidos codificado:", us[:10])
print("México codificado:", mx[:10])
print()

# Combinación de representaciones
mx_us = torchhd.bind(torchhd.inverse(us), mx)
print("=== Combinación de Estados Unidos y México ===")
print("mx_us:", mx_us[:10])
print()

# Consulta: ¿cuál es el “dólar” de México?
usd_of_mex = torchhd.bind(mx_us, united_states_dollar)
print("=== Pregunta: ¿Cuál es el dólar de México? ===")
print("usd_of_mex:", usd_of_mex[:10])
print()

memory = torch.cat([keys, united_states_values, mexico_values], dim=0)
memory_labels = [
    "country", "capital", "currency",
    "united_states", "washington_dc", "united_states_dollar",
    "mexico", "mexico_city", "mexican_peso"
]

similarities = torchhd.cosine_similarity(usd_of_mex, memory)

print("=== Similitud con vectores conocidos ===")
for label, sim in zip(memory_labels, similarities.tolist()):
    print(f"{label:20s}: {sim:.4f}")

best_match_idx = torch.argmax(similarities).item()
best_label = memory_labels[best_match_idx]
print("\n💡 Resultado más probable:", best_label)
