import numpy as np
import cupy as cp
import time

#### CODE généré par GPT, seulement utile pour tester si les calculs GPU fonctionnent

# Définir la taille du tableau
N = 100_000_000

# Création d'un tableau aléatoire sur le CPU
x_cpu = np.random.rand(N)
# Transfert du tableau sur le GPU
x_gpu = cp.asarray(x_cpu)

# Mesure du temps d'exécution sur le CPU
start_cpu = time.time()
result_cpu = np.sum(x_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

# Mesure du temps d'exécution sur le GPU
start_gpu = time.time()
result_gpu = cp.sum(x_gpu)

# Nécessaire de synchroniser pour s'assurer que le calcul GPU est terminé
cp.cuda.Stream.null.synchronize()
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

# Affichage des résultats et des temps d'exécution
print("Résultat CPU :", result_cpu)
print("Temps CPU    : {:.6f} secondes".format(cpu_time))

print("Résultat GPU :", cp.asnumpy(result_gpu))
print("Temps GPU    : {:.6f} secondes".format(gpu_time))
