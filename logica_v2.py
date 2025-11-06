import torch
import torchhd

dimension = 10000

siblings, father, some_1, some_2 = torchhd.random(4, dimension)
A, B, C = torchhd.random(3, dimension)
Juan, Maria, Pedro = torchhd.random(3, dimension)

siblings_head = torchhd.bundle(siblings, torchhd.bind(A, B))
siblings_body = torchhd.bundle(torchhd.bundle(father, torchhd.bind(C, A)), torchhd.bundle(father, torchhd.bind(C, B)))
rule_siblings = torchhd.bundle(siblings_head, siblings_body)

some_1_head = torchhd.bundle(some_1, torchhd.bind(A, B))
some_1_body = torchhd.bundle(torchhd.bundle(some_2, torchhd.bind(C, A)), torchhd.bundle(some_2, torchhd.bind(C, B)))
rule_some_1 = torchhd.bundle(some_1_head, some_1_body)

memory_rules = torchhd.bundle(rule_siblings, rule_some_1)

"""
siblings_head = siblings + A * B
siblings_body = father + C * A + father + C * B 
rule_siblings = siblings + A * B + father + C * A + father + C * B 

memory_rules = siblings + A * B + father + C * A + father + C * B + some_1 + A * B + some_2 + C * A + some_2 + C * B 
"""

fact_1 = torchhd.bundle(father, torchhd.bind(Juan, Maria))
fact_2 = torchhd.bundle(father, torchhd.bind(Juan, Pedro))

memory_facts = torchhd.bundle(fact_1, fact_2)

"""
fact_1 = father + Juan * Maria
fact_2 = father + Juan * Pedro

memory_facts = father + Juan * Maria + father + Juan * Pedro
"""

query = torchhd.bundle(siblings, torchhd.bind(Juan, Maria))

sim = torchhd.cosine_similarity(query, rule_siblings)
print(f"\nSimilaridad:  {sim.item():.4f}")
