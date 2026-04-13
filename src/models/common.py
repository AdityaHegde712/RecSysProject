from src.models.knn import ItemKNN


def build_model(config: dict, num_users: int, num_items: int) -> ItemKNN:
    """Instantiate an ItemKNN model from config.

    config['model']['name'] should be 'itemknn'. Extra keys under
    config['model'] are forwarded as kwargs.
    """
    model_cfg = config["model"]
    name = model_cfg["name"].lower()

    if name == "itemknn":
        k_neighbors = model_cfg.get("k_neighbors", 50)
        return ItemKNN(k_neighbors=k_neighbors)

    raise ValueError(f"Unknown model: {name}. Only 'itemknn' is supported.")
