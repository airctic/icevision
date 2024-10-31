__all__ = ["model"]

from icevision.imports import *
from icevision.utils import *

import yaml
import yolov5
from yolov5.models.yolo import Model
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import check_img_size, intersect_dicts
from icevision.models.ultralytics.yolov5.utils import *
from icevision.models.ultralytics.yolov5.backbones import *

yolo_dir = get_root_dir() / "yolo"
yolo_dir.mkdir(exist_ok=True)


def model(
    backbone: YoloV5BackboneConfig,
    num_classes: int,
    img_size: int,  # must be multiple of 32
    device: Optional[torch.device] = None,
) -> nn.Module:
    model_name = backbone.model_name
    pretrained = backbone.pretrained

    # this is to remove background from ClassMap as discussed
    # here: https://github.com/ultralytics/yolov5/issues/2950
    # and here: https://discord.com/channels/735877944085446747/782062040168267777/836692604224536646
    # so we should pass `num_classes=parser.class_map.num_classes`
    num_classes -= 1

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else device
    )

    if model_name in ["yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
        cfg_filepath = Path(yolov5.__file__).parent / f"models/{model_name}.yaml"
    else:
        cfg_filepath = Path(yolov5.__file__).parent / f"models/hub/{model_name}.yaml"

    if pretrained:
        weights_path = yolo_dir / f"{model_name}.pt"

        with open(Path(yolov5.__file__).parent / "data/hyps/hyp.VOC.yaml") as f:
            hyp = yaml.safe_load(f)

        attempt_download(weights_path)  # download if not found locally
        sys.path.insert(0, str(Path(yolov5.__file__).parent))
        ckpt = torch.load(weights_path, map_location=device)  # load checkpoint
        sys.path.remove(str(Path(yolov5.__file__).parent))
        if hyp.get("anchors"):
            ckpt["model"].yaml["anchors"] = round(hyp["anchors"])  # force autoanchor
        model = Model(cfg_filepath or ckpt["model"].yaml, ch=3, nc=num_classes).to(
            device
        )  # create
        exclude = []  # exclude keys
        state_dict = ckpt["model"].float().state_dict()  # to FP32
        state_dict = intersect_dicts(
            state_dict, model.state_dict(), exclude=exclude
        )  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
    else:
        with open(Path(yolov5.__file__).parent / "data/hyps/hyp.scratch-med.yaml") as f:
            hyp = yaml.safe_load(f)

        model = Model(
            cfg_filepath, ch=3, nc=num_classes, anchors=hyp.get("anchors")
        ).to(
            device
        )  # create

    gs = int(model.stride.max())  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(img_size, gs)  # verify imgsz are gs-multiples

    hyp["box"] *= 3.0 / nl  # scale to layers
    hyp["cls"] *= num_classes / 80.0 * 3.0 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3.0 / nl  # scale to image size and layers
    model.nc = num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    def param_groups_fn(model: nn.Module) -> List[List[nn.Parameter]]:
        spp_index = [
            i + 1
            for i, layer in enumerate(model.model.children())
            if layer._get_name() == "SPPF"
        ][0]
        backbone = list(model.model.children())[:spp_index]
        neck = list(model.model.children())[spp_index:-1]
        head = list(model.model.children())[-1]

        layers = [nn.Sequential(*backbone), nn.Sequential(*neck), nn.Sequential(head)]

        param_groups = [list(group.parameters()) for group in layers]
        check_all_model_params_in_groups2(model.model, param_groups)

        return param_groups

    model.param_groups = MethodType(param_groups_fn, model)

    return model
