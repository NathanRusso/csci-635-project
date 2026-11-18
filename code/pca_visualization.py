from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras


NUM_SAMPLES = 100
MODEL_PATH = "data/635-CNN.keras"


def standard_pca(X):
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    return pca.fit_transform(X_std)

def cnn_embedding_pca(X):
    model = keras.models.load_model(MODEL_PATH)

    # Extract output of second MaxPooling layer
    feature_maps = model.layers[3].output
    embeddings = keras.layers.GlobalAveragePooling2D()(feature_maps)

    embedding_model = keras.Model(
        inputs=model.inputs,
        outputs=embeddings
    )

    # Extract CNN embeddings
    X_images = X.reshape(-1, 700, 500, 1)
    X_embeddings = embedding_model.predict(X_images)

    # Apply PCA
    return standard_pca(X_embeddings)


def visualize_pca(X, y, title):
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
    plt.title(title)
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
    y = np.array([0] * NUM_SAMPLES + [1] * NUM_SAMPLES + [2] * NUM_SAMPLES)

    # Apply standard PCA
    X_pca_raw = standard_pca(X)

    # Apply PCA over CNN embeddings
    X_pca_cnn = cnn_embedding_pca(X)

    # plot samples in PCA space
    visualize_pca(X_pca_raw, y, "PCA of Raw Images")
    visualize_pca(X_pca_cnn, y, "PCA of CNN Embeddings")

    
if __name__ == "__main__":
    main()