__all__ = ["get_files", "get_image_files"]

from icevision.imports import *

# All copied from fastai
def _get_files(p, fs, extensions=None):
    p = Path(p)
    res = [
        p / f
        for f in fs
        if not f.startswith(".")
        and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)
    ]
    return res


def get_files(
    path,
    extensions=None,
    recurse=True,
    folders=None,
    followlinks=True,
    sort: bool = True,
):
    "Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified. From fastai"
    path = Path(path)
    folders = L(folders)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(
            os.walk(path, followlinks=followlinks)
        ):  # returns (dirpath, dirnames, filenames)
            if len(folders) != 0 and i == 0:
                d[:] = [o for o in d if o in folders]
            else:
                d[:] = [o for o in d if not o.startswith(".")]
            if len(folders) != 0 and i == 0 and "." not in folders:
                continue
            res += _get_files(p, f, extensions)
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, f, extensions)

    return L(sorted(res)) if sort else L(res)


image_extensions = set(
    k for k, v in mimetypes.types_map.items() if v.startswith("image/")
)


def get_image_files(path, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified. From fastai"
    return get_files(
        path, extensions=image_extensions, recurse=recurse, folders=folders
    )
