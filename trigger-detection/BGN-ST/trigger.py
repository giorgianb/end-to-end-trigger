import numpy as np
from sklearn.linear_model import LogisticRegression

f = np.load('trigger_data.npz')

p_t = f['p_t']
energy = f['energy']
n_hits = f['n_hits']
trigger = f['trigger']



p_t_model = LogisticRegression().fit(p_t.reshape(-1, 1), trigger)
print(f'p_t_model: ', p_t_model.score(p_t.reshape(-1, 1), trigger))

energy_model = LogisticRegression().fit(energy.reshape(-1, 1), trigger)
print(f'energy_model: ', energy_model.score(energy.reshape(-1, 1), trigger))

n_hits_model = LogisticRegression().fit(n_hits.reshape(-1, 1), trigger)
print(f'n_hits_model: ', n_hits_model.score(n_hits.reshape(-1, 1), trigger))

X = np.stack([p_t, n_hits, energy], axis=-1)
total_model = LogisticRegression().fit(X, trigger)
print(f'total_model: ', total_model.score(X, trigger))
