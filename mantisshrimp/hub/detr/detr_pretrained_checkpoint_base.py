__all__ = ["detr_pretrained_checkpoint_base"]

from mantisshrimp.imports import *


def detr_pretrained_checkpoint_base():
    # load checkpoint and delete head
    url = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"
    checkpoint = torch.hub.load_state_dict_from_url(
        url, progress=False, map_location="cpu"
    )
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]
    save_path = os.path.join(torch.hub._get_torch_home(), "detr-r50_no-class-head.pth")
    torch.save(checkpoint, save_path)
    return save_path
