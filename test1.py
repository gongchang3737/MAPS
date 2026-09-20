import random

w = [0.1,0.2,0.3,0.4]
n = [1,2,3,4]
samples = []
k = 3

for i in range(k):
    sample = random.choices(n, weights = w)
    samples.append(sample[0])
    
    w.remove(w[sample.index(sample[0])])
    n.remove(sample[0]) #放在w.remove之后
    
print(samples)

