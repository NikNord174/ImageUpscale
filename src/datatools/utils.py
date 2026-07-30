def resolve_tuple(*args):
    """OmegaConf resolver so configs can write ${tuple:128,128}."""
    return tuple(args)
