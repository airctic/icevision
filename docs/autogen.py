import os
from pathlib import Path
import shutil

import keras_autodoc

# from keras_autodoc.examples import copy_examples
import tutobooks

PAGES = {
    "faster_rcnn.md": [
        "mantisshrimp.models.rcnn.faster_rcnn.model.model",
        "mantisshrimp.models.rcnn.faster_rcnn.dataloaders.train_dataloader",
        "mantisshrimp.models.rcnn.faster_rcnn.dataloaders.valid_dataloader",
        "mantisshrimp.models.rcnn.faster_rcnn.dataloaders.infer_dataloader",
        "mantisshrimp.models.rcnn.faster_rcnn.dataloaders.build_train_batch",
        "mantisshrimp.models.rcnn.faster_rcnn.dataloaders.build_valid_batch",
        "mantisshrimp.models.rcnn.faster_rcnn.dataloaders.build_infer_batch",
    ],
    "faster_rcnn_fastai.md": [
        "mantisshrimp.models.rcnn.faster_rcnn.fastai.learner.learner",
    ],
    "faster_rcnn_lightning.md": [
        "mantisshrimp.models.rcnn.faster_rcnn.lightning.model_adapter.ModelAdapter",
    ],
    "mask_rcnn.md": [
        "mantisshrimp.models.rcnn.mask_rcnn.model.model",
    ],
}

# aliases_needed = [
#     'tensorflow.keras.callbacks.Callback',
#     'tensorflow.keras.losses.Loss',
#     'tensorflow.keras.metrics.Metric',
#     'tensorflow.data.Dataset'
# ]


ROOT = "https://airctic.github.io/mantisshrimp/"

mantisshrimp_dir = Path(__file__).resolve().parents[1]


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

        github_repo_dir = "airctic/mantisshrimp/blob/master/docs/"
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


def nb_to_md(dest_dir):
    notebooks_dir = mantisshrimp_dir / "notebooks"
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
    examples_dir = mantisshrimp_dir / "examples"
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


def generate(dest_dir):
    template_dir = mantisshrimp_dir / "docs" / "templates"

    # Create dest_dir if doesn't exist
    if os.path.exists(dest_dir):
        print("Removing sources folder:", dest_dir)
        shutil.rmtree(dest_dir)

    os.makedirs(dest_dir)

    doc_generator = keras_autodoc.DocumentationGenerator(
        pages=PAGES,
        project_url='https://github.com/airctic/mantisshrimp/blob/master',
        template_dir=template_dir,
        examples_dir=mantisshrimp_dir / 'examples',
    )
    doc_generator.generate(dest_dir)

    # Auto generate the index.md file using the README.md file and the index.md file in templates folder
    readme = (mantisshrimp_dir / "README.md").read_text()

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
    shutil.copyfile(mantisshrimp_dir / "ABOUT.md", dest_dir / "about.md")
    shutil.copyfile(mantisshrimp_dir / "CONTRIBUTING.md", dest_dir / "contributing.md")
    shutil.copyfile(mantisshrimp_dir / "DOCKER.md", dest_dir / "docker.md")
    shutil.copyfile(mantisshrimp_dir / "INSTALL.md", dest_dir / "install.md")
    shutil.copyfile(
        mantisshrimp_dir / "README_MKDOCS.md", dest_dir / "readme_mkdocs.md"
    )
    shutil.copyfile(
        mantisshrimp_dir / "CHANGING-THE-COLORS.md", dest_dir / "changing-the-colors.md"
    )
    shutil.copyfile(mantisshrimp_dir / "DEPLOYMENT.md", dest_dir / "deployment.md")

    # Copy images folder from the template folder to the destination folder
    template_images_dir = Path(template_dir) / "images"
    print("Template folder: ", template_images_dir)
    dest_images_dir = Path(dest_dir) / "images"

    if not os.path.exists(dest_images_dir):
        os.makedirs(dest_images_dir)

    if os.path.exists(template_images_dir):
        for fname in os.listdir(template_images_dir):
            src = Path(template_images_dir) / fname
            target = Path(dest_images_dir) / fname
            print("copy", src, "to", target)
            shutil.copyfile(src, target)

    # Generate .md files form Jupyter Notebooks located in the /ipynb folder
    nb_to_md(dest_dir)

    # Generate .md files form python files located in the /examples folder
    examples_to_md(dest_dir)


if __name__ == "__main__":
    generate(mantisshrimp_dir / "docs" / "sources")
