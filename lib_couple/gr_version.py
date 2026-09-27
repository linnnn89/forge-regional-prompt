from gradio import __version__ as gradio_version

is_neo: bool = not gradio_version.startswith("3")


def js(func: str) -> dict:
    return {("js" if is_neo else "_js"): func}
