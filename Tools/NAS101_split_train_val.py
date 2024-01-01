import numpy as np
import h5py


def denoise_nasbench(metrics, threshold=0.8):
    val_metrics = metrics[:, -1, :, -1, 2]
    print(len(val_metrics))
    index = np.where(val_metrics[:, 0] > threshold)
    print(len(index[0]))
    return index[0]


with h5py.File("./Datas/nasbench.hdf5", mode="r") as f:
    total_count = len(f["hash"][()])
    print(total_count)
    metrics = f["metrics"][()]
random_state = np.random.RandomState(0)
result = dict()
for n_samples in [172, 334, 860]:
    split = random_state.permutation(total_count)[:n_samples]
    result[str(n_samples)] = split

# >91
valid91 = denoise_nasbench(metrics, threshold=0.91)
for n_samples in [172, 334, 860]:
    result["91-" + str(n_samples)] = np.intersect1d(result[str(n_samples)], valid91)
result["denoise-91"] = valid91

result["denoise-80"] = denoise_nasbench(metrics)
print(len(result['172']))
np.savez("./Datas/NAS_101train.npz", **result)
