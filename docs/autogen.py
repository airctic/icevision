import os
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

import keras_autodoc

# from keras_autodoc.examples import copy_examples
import tutobooks

PAGES = {
    "parser.md": [
        "icevision.parsers.Parser",
        "icevision.parsers.Parser.parse",
        "icevision.parsers.FasterRCNN",
        "icevision.parsers.MaskRCNN",
        "icevision.parsers.mixins.ImageidMixin",
        "icevision.parsers.mixins.FilepathMixin",
        "icevision.parsers.mixins.SizeMixin",
        "icevision.parsers.mixins.LabelsMixin",
        "icevision.parsers.mixins.BBoxesMixin",
        "icevision.parsers.mixins.MasksMixin",
        "icevision.parsers.mixins.AreasMixin",
        "icevision.parsers.mixins.IsCrowdsMixin",
    ],
    "dataset.md": [
        "icevision.data.dataset.Dataset",
        "icevision.data.dataset.Dataset.from_images",
    ],
    "albumentations_tfms.md": [
        "icevision.tfms.albumentations.aug_tfms",
        "icevision.tfms.albumentations.Adapter",
    ],
    # "coco_metric.md": [
    # "icevision.metrics.coco_metric.coco_metric.COCOMetric",
    # "icevision.metrics.coco_metric.coco_metric.COCOMetricType",
    # ],
    "data_splits.md": [
        "icevision.data.DataSplitter",
        "icevision.data.RandomSplitter",
        "icevision.data.FixedSplitter",
        "icevision.data.SingleSplitSplitter",
    ],
    "faster_rcnn.md": [
        "icevision.models.torchvision.faster_rcnn.model.model",
        "icevision.models.torchvision.faster_rcnn.dataloaders.train_dl",
        "icevision.models.torchvision.faster_rcnn.dataloaders.valid_dl",
        "icevision.models.torchvision.faster_rcnn.dataloaders.infer_dl",
        "icevision.models.torchvision.faster_rcnn.dataloaders.build_train_batch",
        "icevision.models.torchvision.faster_rcnn.dataloaders.build_valid_batch",
        "icevision.models.torchvision.faster_rcnn.dataloaders.build_infer_batch",
    ],
    "faster_rcnn_fastai.md": [
        "icevision.models.torchvision.faster_rcnn.fastai.learner.learner",
    ],
    "faster_rcnn_lightning.md": [
        "icevision.models.torchvision.faster_rcnn.lightning.model_adapter.ModelAdapter",
    ],
    "mask_rcnn.md": [
        "icevision.models.torchvision.mask_rcnn.model.model",
        "icevision.models.torchvision.mask_rcnn.dataloaders.train_dl",
        "icevision.models.torchvision.mask_rcnn.dataloaders.valid_dl",
        "icevision.models.torchvision.mask_rcnn.dataloaders.infer_dl",
        "icevision.models.torchvision.mask_rcnn.dataloaders.build_train_batch",
        "icevision.models.torchvision.mask_rcnn.dataloaders.build_valid_batch",
        "icevision.models.torchvision.mask_rcnn.dataloaders.build_infer_batch",
    ],
    "mask_rcnn_fastai.md": [
        "icevision.models.torchvision.mask_rcnn.fastai.learner.learner",
    ],
    "mask_rcnn_lightning.md": [
        "icevision.models.torchvision.mask_rcnn.lightning.model_adapter.ModelAdapter",
    ],
    "efficientdet.md": [
        "icevision.models.efficientdet.model.model",
        "icevision.models.efficientdet.dataloaders.train_dl",
        "icevision.models.efficientdet.dataloaders.valid_dl",
        "icevision.models.efficientdet.dataloaders.infer_dl",
        "icevision.models.efficientdet.dataloaders.build_train_batch",
        "icevision.models.efficientdet.dataloaders.build_valid_batch",
        "icevision.models.efficientdet.dataloaders.build_infer_batch",
    ],
    "efficientdet_fastai.md": [
        "icevision.models.efficientdet.fastai.learner.learner",
    ],
    "efficientdet_lightning.md": [
        "icevision.models.efficientdet.lightning.model_adapter.ModelAdapter",
    ],
}

# aliases_needed = [
#     'tensorflow.keras.callbacks.Callback',
#     'tensorflow.keras.losses.Loss',
#     'tensorflow.keras.metrics.Metric',
#     'tensorflow.data.Dataset'
# ]


ROOT = "https://airctic.github.io/icevision/"

icevision_dir = Path(__file__).resolve().parents[1]
print("icevision_dir: ", icevision_dir)


# From keras_autodocs
def copy_examples(examples_dir, destination_dir):
    """Copy the examples directory in the documentation.

    Prettify files by extracting the docstrings written in Markdown.
    """
    Path(destination_dir).mkdir(exist_ok=True)
    for file in os.listdir(examples_dir):
        if not file.endswith(".py"):
            continue
        module_path = os.path.join(examples_dir, file)
        docstring, starting_line = get_module_docstring(module_path)
        print("dostring", docstring)
        print("starting_line", starting_line)
        destination_file = os.path.join(destination_dir, file[:-2] + "md")
        with open(destination_file, "w+", encoding="utf-8") as f_out, open(
            examples_dir / file, "r+", encoding="utf-8"
        ) as f_in:

            if docstring:
                f_out.write(docstring + "\n\n")

            # skip docstring
            for _ in range(starting_line + 2):
                next(f_in)

            f_out.write("```python\n")
            # next line might be empty.
            line = next(f_in)
            if line != "\n":
                f_out.write(line)

            # copy the rest of the file.
            for line in f_in:
                f_out.write(line)
            f_out.write("\n```")


def get_module_docstring(filepath):
    """Extract the module docstring.

    Also finds the line at which the docstring ends.
    """
    co = compile(open(filepath, encoding="utf-8").read(), filepath, "exec")
    if co.co_consts and isinstance(co.co_consts[0], str):
        docstring = co.co_consts[0]
    else:
        print("Could not get the docstring from " + filepath)
        docstring = ""
    return docstring, co.co_firstlineno


# end


def py_to_nb_md(dest_dir):
    for file_path in os.listdir("py/"):
        dir_path = "py"
        file_name = file_path
        py_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != ".py":
            continue

        nb_path = os.path.join("ipynb", file_name_no_ext + ".ipynb")
        md_path = os.path.join(dest_dir, "tutorial", file_name_no_ext + ".md")

        tutobooks.py_to_md(py_path, nb_path, md_path, "templates/img")

        github_repo_dir = "airctic/icevision/blob/master/docs/"
        with open(md_path, "r") as md_file:
            button_lines = [
                ":material-link: "
                "[**View in Colab**](https://colab.research.google.com/github/"
                + github_repo_dir
                + "ipynb/"
                + file_name_no_ext
                + ".ipynb"
                + ")   &nbsp; &nbsp;"
                # + '<span class="k-dot">•</span>'
                + ":octicons-octoface: "
                "[**GitHub source**](https://github.com/"
                + github_repo_dir
                + "py/"
                + file_name_no_ext
                + ".py)",
                "\n",
            ]
            md_content = "".join(button_lines) + "\n" + md_file.read()

        with open(md_path, "w") as md_file:
            md_file.write(md_content)


def nb_to_md(src_dir, nb_folder, dest_dir):
    notebooks_dir = src_dir / nb_folder
    print("Notebooks folder: ", notebooks_dir)

    for file_path in os.listdir(notebooks_dir):
        dir_path = notebooks_dir
        file_name = file_path
        nb_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != ".ipynb":
            continue

        # md_path = os.path.join(dest_dir, 'tutorial', file_name_no_ext + '.md')
        md_path = os.path.join(dest_dir, file_name_no_ext + ".md")
        images_path = "images"

        tutobooks.nb_to_md(nb_path, md_path, images_path)


def examples_to_md(dest_dir):
    examples_dir = icevision_dir / "examples"
    print("Examples folder: ", examples_dir)

    for file_path in os.listdir(examples_dir):
        dir_path = examples_dir
        file_name = file_path
        nb_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != ".py":
            continue

        # md_path = os.path.join(dest_dir, 'tutorial', file_name_no_ext + '.md')
        md_path = os.path.join(dest_dir, file_name_no_ext + ".md")

        copy_examples(examples_dir, dest_dir / "examples")


def generate(dest_dir: Path):
    template_dir = icevision_dir / "docs" / "templates"
    template_images_dir = Path(template_dir) / "images"

    # Create dest_dir if doesn't exist
    if os.path.exists(dest_dir):
        print("Removing sources folder:", dest_dir)
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    # Copy images folder from root folder to the template images folder
    copy_tree(str(icevision_dir / "images"), str(template_images_dir))

    # Generate APIs Documentation
    doc_generator = keras_autodoc.DocumentationGenerator(
        pages=PAGES,
        project_url="https://github.com/airctic/icevision/blob/master",
        template_dir=template_dir,
        examples_dir=icevision_dir / "examples",
    )
    doc_generator.generate(dest_dir)

    # Copy CNAME file
    shutil.copyfile(icevision_dir / "CNAME", dest_dir / "CNAME")

    # Copy web manifest
    shutil.copyfile("manifest.webmanifest", dest_dir / "manifest.webmanifest")

    # Auto generate the index.md file using the README.md file and the index.md file in templates folder
    readme = (icevision_dir / "README.md").read_text()

    # Search for the beginning and the end of the installation procedure to hide in Docs to avoid duplication
    start = readme.find("<!-- Not included in docs - start -->")
    end = readme.find("<!-- Not included in docs - end -->")
    print("\nSTART: ", start)
    print("END: ", end, "\n")
    readme = readme.replace(readme[start:end], "")
    index = (template_dir / "index.md").read_text()
    index = index.replace("{{autogenerated}}", readme[readme.find("##") :])
    (dest_dir / "index.md").write_text(index, encoding="utf-8")

    # Copy static .md files from the root folder
    shutil.copyfile(icevision_dir / "CONTRIBUTING.md", dest_dir / "contributing.md")
    shutil.copyfile(
        icevision_dir / "CODE_OF_CONDUCT.md", dest_dir / "code_of_conduct.md"
    )

    # Copy static .md files from the docs folder
    shutil.copyfile(icevision_dir / "docs/INSTALL.md", dest_dir / "install.md")
    shutil.copyfile(
        icevision_dir / "docs/HOW-TO.md",
        dest_dir / "how-to.md",
    )
    shutil.copyfile(icevision_dir / "docs/ABOUT.md", dest_dir / "about.md")

    shutil.copyfile(icevision_dir / "docs/README.md", dest_dir / "readme_mkdocs.md")

    shutil.copyfile(
        icevision_dir / "docs/CHANGING-THE-COLORS.md",
        dest_dir / "changing_the_colors.md",
    )

    shutil.copyfile(icevision_dir / "docs/DEPLOYMENT.md", dest_dir / "deployment.md")

    # Copy static .md files from the other folders
    shutil.copyfile(
        icevision_dir / "icevision/models/README.md",
        dest_dir / "model_comparison.md",
    )

    shutil.copyfile(
        icevision_dir / "icevision/models/efficientdet/README.md",
        dest_dir / "model_efficientdet.md",
    )

    shutil.copyfile(
        icevision_dir / "icevision/models/torchvision/faster_rcnn/README.md",
        dest_dir / "model_faster_rcnn.md",
    )

    shutil.copyfile(
        icevision_dir / "icevision/backbones/backbones_effecientdet.md",
        dest_dir / "backbones_effecientdet.md",
    )

    shutil.copyfile(
        icevision_dir / "icevision/backbones/backbones_faster_mask_rcnn.md",
        dest_dir / "backbones_faster_mask_rcnn.md",
    )

    shutil.copyfile(
        icevision_dir / "icevision/tfms/README.md",
        dest_dir / "albumentations.md",
    )

    # Copy .md examples files to destination examples folder
    # Copy css folder
    copy_tree(str(icevision_dir / "examples"), str(dest_dir / "examples"))

    # Copy images folder from the template folder to the destination folder
    print("Template folder: ", template_images_dir)
    dest_images_dir = Path(dest_dir) / "images"

    # Copy images folder
    copy_tree(str(template_images_dir), str(dest_images_dir))

    # Copy css folder
    copy_tree(str(icevision_dir / "docs/css"), str(dest_dir / "css"))

    # Copy js folder
    copy_tree(str(icevision_dir / "docs/js"), str(dest_dir / "js"))

    # Generate .md files form Jupyter Notebooks located in the /notebooks folder
    nb_to_md(icevision_dir, "notebooks", dest_dir)

    # Generate .md files form Jupyter Notebooks located in the /deployment folder
    nb_to_md(icevision_dir / "docs", "deployment", dest_dir)

    # Generate .md files form python files located in the /examples folder
    # examples_to_md(dest_dir)


if __name__ == "__main__":
    generate(icevision_dir / "docs" / "sources")
