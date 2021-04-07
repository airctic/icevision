import os
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

import keras_autodoc

# from keras_autodoc.examples import copy_examples
import tutobooks
from loguru import logger

PAGES = {
    "parser.md": [
        "icevision.parsers.Parser",
        "icevision.parsers.Parser.parse",
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
        "icevision.models.ross.efficientdet.model.model",
        "icevision.models.ross.efficientdet.dataloaders.train_dl",
        "icevision.models.ross.efficientdet.dataloaders.valid_dl",
        "icevision.models.ross.efficientdet.dataloaders.infer_dl",
        "icevision.models.ross.efficientdet.dataloaders.build_train_batch",
        "icevision.models.ross.efficientdet.dataloaders.build_valid_batch",
        "icevision.models.ross.efficientdet.dataloaders.build_infer_batch",
    ],
    "efficientdet_fastai.md": [
        "icevision.models.ross.efficientdet.fastai.learner.learner",
    ],
    "efficientdet_lightning.md": [
        "icevision.models.ross.efficientdet.lightning.model_adapter.ModelAdapter",
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

        from_to = f"{file} -> {destination_file}"
        logger.opt(colors=True).log(
            "INFO",
            "️<green><bold>Copying Examples: {}</></>",
            from_to,
        )


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

        github_repo_dir = "airctic/icedata/blob/master/docs/"
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
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>Notebooks folder: {}</></>",
        notebooks_dir,
    )

    for file_path in os.listdir(notebooks_dir):
        dir_path = notebooks_dir
        file_name = file_path
        nb_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != ".ipynb":
            continue

        # md_path = os.path.join(dest_dir, 'tutorial', file_name_no_ext + '.md')
        file_name_md = file_name_no_ext + ".md"
        # md_path = os.path.join(dest_dir, file_name_md)
        md_path = os.path.join(dest_dir, file_name_no_ext + ".md")
        images_path = "images"

        tutobooks.nb_to_md(nb_path, md_path, images_path)
        from_to = f"{file_name} -> {file_name_md}"
        logger.opt(colors=True).log(
            "INFO",
            "️<green><bold>Converting to Notebook: {}</></>",
            from_to,
        )


def examples_to_md(dest_dir):
    examples_dir = icevision_dir / "examples"
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>Examples folder: {}</></>",
        examples_dir,
    )

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

        from_to = f"{nb_path} -> {md_path}"
        logger.opt(colors=True).log(
            "INFO",
            "️<green><bold>Copying Examples: {}</></>",
            from_to,
        )


def generate(dest_dir: Path):
    template_dir = icevision_dir / "docs" / "templates"
    template_images_dir = Path(template_dir) / "images"

    # Create dest_dir if doesn't exist
    if os.path.exists(dest_dir):
        print("Removing sources folder:", dest_dir)
        logger.opt(colors=True).log(
            "INFO",
            "️<magenta><bold>\nRemoving sources folder: {}</></>",
            dest_dir,
        )
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    # Copy images folder from root folder to the template images folder
    copy_tree(str(icevision_dir / "images"), str(template_images_dir))
    from_to = f"root/images -> docs/images"
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>\nCopying images folder: {}</></>",
        from_to,
    )

    # Generate APIs Documentation
    doc_generator = keras_autodoc.DocumentationGenerator(
        pages=PAGES,
        project_url="https://github.com/airctic/icedata/blob/master",
        template_dir=template_dir,
        examples_dir=icevision_dir / "examples",
    )
    doc_generator.generate(dest_dir)

    # Copy CNAME file
    shutil.copyfile(icevision_dir / "CNAME", dest_dir / "CNAME")

    # Copy web manifest
    shutil.copyfile("manifest.webmanifest", dest_dir / "manifest.webmanifest")
    from_to = f"root/manifest.webmanifest -> docs/manifest.webmanifest"
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>\nCopying webmanifest file: {}</></>",
        from_to,
    )

    # Auto generate the index.md file using the README.md file and the index.md file in templates folder
    readme = (icevision_dir / "README.md").read_text()

    # Search for the beginning and the end of the installation procedure to hide in Docs to avoid duplication
    start = readme.find("<!-- Not included in docs - start -->")
    end = readme.find("<!-- Not included in docs - end -->")

    readme = readme.replace(readme[start:end], "")
    index = (template_dir / "index.md").read_text()
    index = index.replace("{{autogenerated}}", readme[readme.find("##") :])
    (dest_dir / "index.md").write_text(index, encoding="utf-8")

    # Copy static .md files from the root folder
    dir_to_search = icevision_dir
    fnamelist = [
        filename for filename in os.listdir(dir_to_search) if filename.endswith(".md")
    ]
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>\nCopying .md files root folder: {}</></>",
        fnamelist,
    )

    for fname in fnamelist:
        fname_src = icevision_dir / fname
        fname_dst = dest_dir / fname.lower()
        shutil.copyfile(fname_src, fname_dst)
        from_to = f"{fname} -> {fname.lower()}"
        logger.opt(colors=True).log(
            "INFO",
            "️<light-blue><bold>file: {}</></>",
            from_to,
        )

    # Copy static .md files from the docs folder
    dir_to_search = icevision_dir / "docs"
    fnamelist = [
        filename for filename in os.listdir(dir_to_search) if filename.endswith(".md")
    ]
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>\nCopying .md files from the docs folder: {}</></>",
        fnamelist,
    )
    for fname in fnamelist:
        fname_src = dir_to_search / fname
        fname_dst = dest_dir / fname.lower()
        shutil.copyfile(fname_src, fname_dst)
        from_to = f"{fname} -> {fname.lower()}"
        logger.opt(colors=True).log(
            "INFO",
            "️<light-blue><bold>Copying files: {}</></>",
            from_to,
        )

    # Copy images folder from the template folder to the destination folder
    # print("Template folder: ", template_images_dir)
    dest_images_dir = Path(dest_dir) / "images"

    # Copy images folder
    copy_tree(str(template_images_dir), str(dest_images_dir))
    from_to = f"{template_images_dir} -> {dest_images_dir}"
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>Copying Images: {}</></>",
        from_to,
    )

    # Copy css folder
    css_dir_src = str(icevision_dir / "docs/css")
    css_dir_dest = str(str(dest_dir / "css"))
    copy_tree(css_dir_src, css_dir_dest)
    from_to = f"{css_dir_src} -> {css_dir_dest}"
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>Copying CSS files: {}</></>",
        from_to,
    )

    # Copy js folder
    # copy_tree(str(icevision_dir / "docs/js"), str(dest_dir / "js"))
    js_dir_src = str(icevision_dir / "docs/js")
    js_dir_dest = str(str(dest_dir / "js"))
    copy_tree(js_dir_src, js_dir_dest)
    from_to = f"{js_dir_src} -> {js_dir_dest}"
    logger.opt(colors=True).log(
        "INFO",
        "️<green><bold>Copying JS files: {}</></>",
        from_to,
    )

    # Generate .md files form Jupyter Notebooks located in the /notebooks folder
    nb_to_md(icevision_dir, "notebooks", dest_dir)

    # Generate .md files form Jupyter Notebooks located in the /deployment folder
    nb_to_md(icevision_dir / "docs", "deployment", dest_dir)

    # albumentations
    shutil.copyfile(
        icevision_dir / "icevision/tfms/README.md",
        dest_dir / "albumentations.md",
    )

    # Models
    shutil.copyfile(
        icevision_dir / "icevision/models/README.md",
        dest_dir / "models.md",
    )

    # Backbones
    shutil.copyfile(
        icevision_dir / "icevision/backbones/README.md",
        dest_dir / "backbones.md",
    )


if __name__ == "__main__":
    generate(icevision_dir / "docs" / "sources")
