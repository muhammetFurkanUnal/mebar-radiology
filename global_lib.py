from pathlib import Path

def get_root_folder() -> str:
    return Path.cwd()

def get_abs_path(path:str|Path) -> Path:
    path = Path(path)
    return get_root_folder() / path