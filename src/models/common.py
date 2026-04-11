from src.models.gmf import GMF


def build_model(config: dict, num_users: int, num_items: int) -> GMF:
    """Instantiate a GMF model from config.

    config['model']['name'] should be 'gmf'. Extra keys under
    config['model'] are forwarded as kwargs.
    """
    model_cfg = config["model"]
    name = model_cfg["name"].lower()

    if name == "gmf":
        embed_dim = model_cfg.get("embed_dim",
                                  model_cfg.get("embedding_dim", 64))
        return GMF(num_users, num_items, embed_dim=embed_dim)

    raise ValueError(f"Unknown model: {name}. Only 'gmf' is supported.")
