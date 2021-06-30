def make_slug(class_name: str, probability: float) -> str:
    persents = round(probability)
    return f"{class_name} {persents}"
