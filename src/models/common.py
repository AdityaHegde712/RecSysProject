"""Model factory -- instantiate any supported model from config."""

from src.models.knn import ItemKNN
from src.models.gmf import GMF
from src.models.popularity import PopularityBaseline


def build_model(config: dict, num_users: int, num_items: int):
    """Instantiate a model from config['model']['name']."""
    model_cfg = config["model"]
    name = model_cfg["name"].lower()

    if name == "itemknn":
        k = model_cfg.get("k_neighbors", model_cfg.get("k", 50))
        return ItemKNN(k=k, n_users=num_users, n_items=num_items)

    if name == "gmf":
        embed_dim = model_cfg.get("embedding_dim", model_cfg.get("embed_dim", 64))
        return GMF(num_users, num_items, embed_dim=embed_dim)

    if name == "popularity":
        return PopularityBaseline(n_items=num_items)

    raise ValueError(f"Unknown model: {name}. Supported: itemknn, gmf, popularity")
