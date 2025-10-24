import torch
import torchhd

dimension = 10000

species, habitat, diet = torchhd.random(3, dimension)

# Leon
lion, savanna, carnivore = torchhd.random(3, dimension)

lion_attributes = torchhd.bundle(
    torchhd.bind(habitat, savanna),
    torchhd.bind(diet, carnivore)
)

lion_record = torchhd.bind(torchhd.bind(species, lion), lion_attributes)

turtle, ocean, herbivore = torchhd.random(3, dimension)

turtle_attributes = torchhd.bundle(
    torchhd.bind(habitat, ocean),
    torchhd.bind(diet, herbivore)
)

turtle_record = torchhd.bind(torchhd.bind(species, turtle), turtle_attributes)

memory_hd = torchhd.bundle(lion_record, turtle_record)

print("¿Cuál es la dieta de turtle?")

# memory * (species * turtle) ≈ turtle_attributes
query_key = torchhd.bind(species, turtle)
turtle_attrs = torchhd.bind(memory_hd, query_key)

# turtle_attrs * diet ≈ herbivore
result = torchhd.bind(turtle_attrs, diet)

values = torch.stack([carnivore, herbivore, lion, turtle, savanna, ocean])
labels = ["carnivore", "herbivore", "lion", "turtle", "savanna", "ocean"]

similarities = torchhd.cosine_similarity(result, values)

print("\nSimilaridades:")
for label, sim in zip(labels, similarities.tolist()):
    print(f"{label:20s}: {sim:.4f}")

best_match_idx = torch.argmax(similarities).item()
best_label = labels[best_match_idx]
print("\n💡 Resultado más probable:", best_label)

'''
lion_attributes = (habitat * savanna + diet * carnivore)
lion_record = ((species * lion) * lion_attributes)
lion_record = ((species * lion) * (habitat * savanna + diet * carnivore))
lion_record = ((species * lion) * (habitat * savanna) + (species * lion) * (diet * carnivore))

turtle_attributes = (habitat * ocean + diet * herbivore)
turtle_record = ((species * turtle) * turtle_attributes)
turtle_record = ((species * turtle) * (habitat * ocean + diet * herbivore))
turtle_record = ((species * turtle) * (habitat * ocean) + (species * turtle) * (diet * herbivore))

memory_hd = lion_record + turtle_record
memory_hd = ((species * lion) * (habitat * savanna) + (species * lion) * (diet * carnivore)) + ((species * turtle) * (habitat * ocean) + (species * turtle) * (diet * herbivore))
memory_hd = (species * lion * habitat * savanna) + (species * lion * diet * carnivore) + (species * turtle * habitat * ocean) + (species * turtle * diet * herbivore)

query_key = species * turtle

turtle_attrs = memory_hd * query_key
turtle_attrs = memory_hd * (species * turtle)
turtle_attrs = ((species * lion * habitat * savanna) + (species * lion * diet * carnivore) + (species * turtle * habitat * ocean) + (species * turtle * diet * herbivore)) * (species * turtle)
turtle_attrs = (species * lion * habitat * savanna) * (species * turtle) + (species * lion * diet * carnivore) * (species * turtle) + (species * turtle * habitat * ocean) * (species * turtle) + (species * turtle * diet * herbivore) * (species * turtle)
turtle_attrs = (lion * habitat * savanna * turtle) + (lion * diet * carnivore * turtle) + (habitat * ocean) + (diet * herbivore)

result = turtle_attrs * diet
result = ((lion * habitat * savanna * turtle) + (lion * diet * carnivore * turtle) + (habitat * ocean) + (diet * herbivore)) * diet
result = ((lion * habitat * savanna * turtle) * diet + (lion * diet * carnivore * turtle) * diet + (habitat * ocean) * diet + (diet * herbivore) * diet
result = (lion * habitat * savanna * turtle * diet) + (lion * carnivore * turtle) + (habitat * ocean * diet) + (herbivore)
'''