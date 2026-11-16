from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


NUM_SAMPLES = 100 


def visualize_pca(X, y):
    class_names = ["Magic: The Gathering", "Pokemon", "Yu-Gi-Oh"]
    colors = ["red", "blue", "green"]

    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        mask = (y == i)

        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[i],
            label=class_name,
            alpha=0.65,
            s=40
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Trading Card Image Features")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()


def main():
    # get data samples from each class
    full_mtg = np.memmap(
        "data/validation/validation_mtg.dat",
        dtype=np.float32,
        mode="r",
        shape=(2335,350_000),
    )
    full_pkm = np.memmap(
        "data/validation/validation_pkm.dat",
        dtype=np.float32,
        mode="r",
        shape=(2335,350_000),
    )
    full_ygo = np.memmap(
        "data/validation/validation_ygo.dat",
        dtype=np.float32,
        mode="r",
        shape=(2335,350_000),
    )

    rng = np.random.default_rng()
    random_idxs = rng.choice(len(full_mtg), size=NUM_SAMPLES, replace=False)

    mtg_samples = full_mtg[random_idxs]
    pkm_samples = full_pkm[random_idxs]
    ygo_samples = full_ygo[random_idxs]

    # create X and y using random samples
    X = np.concatenate([mtg_samples, pkm_samples, ygo_samples], axis=0)
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # Apply PCA
    # X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_std)
    X_pca = pca.fit_transform(X)    # trying no standardization, since we normalized already

    # plot samples in PCA space
    visualize_pca(X_pca, y)

    
if __name__ == "__main__":
    main()