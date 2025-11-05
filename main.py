import torch
import torchhd

dimension = 10000

keys = torchhd.random(3, dimension)
country, capital, currency = keys

united_states, mexico = torchhd.random(2, dimension)
washington_dc, mexico_city = torchhd.random(2, dimension)
united_states_dollar, mexican_peso = torchhd.random(2, dimension)

# Agrupar valores de cada país
united_states_values = torch.stack([united_states, washington_dc, united_states_dollar])
mexico_values = torch.stack([mexico, mexico_city, mexican_peso])

# Codificar (binding + bundling)
us = torchhd.hash_table(keys, united_states_values)
mx = torchhd.hash_table(keys, mexico_values)

# Combinación de representaciones
memory = torchhd.bind(us, mx)

"""
(country * united_states + capital * washington_dc + currency * united_states_dollar) * (country * mexico + capital * mexico_city + currency * mexican_peso)

(country * united_states + capital * washington_dc + currency * united_states_dollar) * (country * mexico + capital * mexico_city + currency * mexican_peso)

country * united_states * country * mexico 
+ country * united_states * capital * mexico_city
+ country * united_states * currency * mexican_peso
+ capital * washington_dc * country * mexico 
+ capital * washington_dc * capital * mexico_city
+ capital * washington_dc * currency * mexican_peso
+ currency * united_states_dollar * country * mexico 
+ currency * united_states_dollar * capital * mexico_city
+ currency * united_states_dollar * currency * mexican_peso

united_states * mexico 
+ country * united_states * capital * mexico_city
+ country * united_states * currency * mexican_peso
+ capital * washington_dc * country * mexico 
+washington_dc * mexico_city
+ capital * washington_dc * currency * mexican_peso
+ currency * united_states_dollar * country * mexico 
+ currency * united_states_dollar * capital * mexico_city
+ united_states_dollar * mexican_peso
"""

# Consulta: ¿cuál es el “dólar” de México?
usd_of_mex = torchhd.bind(memory, united_states_dollar)
print("¿Cuál es el dólar de México?")

values = torch.cat([keys, united_states_values, mexico_values], dim=0)
labels = [
    "country", "capital", "currency",
    "united_states", "washington_dc", "united_states_dollar",
    "mexico", "mexico_city", "mexican_peso"
]

similarities = torchhd.cosine_similarity(usd_of_mex, values)

print("=== Similitud con vectores conocidos ===")
for label, sim in zip(labels, similarities.tolist()):
    print(f"{label:20s}: {sim:.4f}")

best_match_idx = torch.argmax(similarities).item()
best_label = labels[best_match_idx]
print("\n💡 Resultado más probable:", best_label)
