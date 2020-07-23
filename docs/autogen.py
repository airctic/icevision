import os
from pathlib import Path
import shutil

import keras_autodoc
import tutobooks

PAGES = {
    'model_faster_rcnn.md': [
        'mantisshrimp.MantisFasterRCNN',
        'mantisshrimp.MantisFasterRCNN.predict',
        'mantisshrimp.MantisFasterRCNN.param_groups',
        'mantisshrimp.MantisFasterRCNN.convert_raw_prediction',
        'mantisshrimp.MantisFasterRCNN.build_training_sample',
    ],
    'model_mask_rcnn.md': [
        'mantisshrimp.MantisMaskRCNN',
        'mantisshrimp.MantisMaskRCNN.predict',
        'mantisshrimp.MantisMaskRCNN.param_groups',
        'mantisshrimp.MantisMaskRCNN.convert_raw_prediction',
        'mantisshrimp.MantisMaskRCNN.build_training_sample',
    ],
}

# aliases_needed = [
#     'tensorflow.keras.callbacks.Callback',
#     'tensorflow.keras.losses.Loss',
#     'tensorflow.keras.metrics.Metric',
#     'tensorflow.data.Dataset'
# ]


ROOT = 'https://lgvaz.github.io/mantisshrimp/'

mantisshrimp_dir = Path(__file__).resolve().parents[1]


def py_to_nb_md(dest_dir):
    for file_path in os.listdir('py/'):
        dir_path = 'py'
        file_name = file_path
        py_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != '.py':
            continue

        nb_path = os.path.join('ipynb',  file_name_no_ext + '.ipynb')
        md_path = os.path.join(dest_dir, 'tutorial', file_name_no_ext + '.md')

        tutobooks.py_to_md(py_path, nb_path, md_path, 'templates/img')

        github_repo_dir = 'lgvaz/mantisshrimp/blob/master/docs/'
        with open(md_path, 'r') as md_file:
            button_lines = [
                ':material-link: '
                "[**View in Colab**](https://colab.research.google.com/github/"
                + github_repo_dir
                + "ipynb/"
                + file_name_no_ext + ".ipynb"
                + ")   &nbsp; &nbsp;"
                # + '<span class="k-dot">•</span>'
                + ':octicons-octoface: '
                "[**GitHub source**](https://github.com/" + github_repo_dir + "py/"
                + file_name_no_ext + ".py)",
                "\n",
            ]
            md_content = ''.join(button_lines) + '\n' + md_file.read()

        with open(md_path, 'w') as md_file:
            md_file.write(md_content)

def nb_to_md(dest_dir):
    for file_path in os.listdir('ipynb/'):
        dir_path = 'ipynb'
        file_name = file_path
        nb_path = os.path.join(dir_path, file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]

        if ext != '.ipynb':
            continue

        # md_path = os.path.join(dest_dir, 'tutorial', file_name_no_ext + '.md')
        md_path = os.path.join(dest_dir, file_name_no_ext + '.md')
        images_path = 'images'  
              
        tutobooks.nb_to_md(nb_path, md_path, images_path)


def generate(dest_dir):
    template_dir = mantisshrimp_dir / 'docs' / 'templates'

    # Create dest_dir if doesn't exist
    if os.path.exists(dest_dir):
        print("Removing sources folder:", dest_dir)
        shutil.rmtree(dest_dir)
        
    os.makedirs(dest_dir)
    
    # doc_generator = keras_autodoc.DocumentationGenerator(
    #     PAGES,
    #     'https://github.com/lgvaz/mantisshrimp/blob/master',
    #     template_dir,
    #     mantisshrimp_dir / 'examples',
    # )
    # doc_generator.generate(dest_dir)
    
    # Auto generate the index.md file using the README.md file and the index.md file in templates folder
    readme = (mantisshrimp_dir / 'README.md').read_text()
    index = (template_dir / 'index.md').read_text()
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    (dest_dir / 'index.md').write_text(index, encoding='utf-8')
    
    # Copy static .md files from the root folder
    shutil.copyfile(mantisshrimp_dir / 'ABOUT.md',
                    dest_dir / 'about.md')
    shutil.copyfile(mantisshrimp_dir / 'CONTRIBUTING.md',
                    dest_dir / 'contributing.md')
    shutil.copyfile(mantisshrimp_dir / 'DOCKER.md',
                    dest_dir / 'docker.md')
    shutil.copyfile(mantisshrimp_dir / 'INSTALL.md',
                    dest_dir / 'install.md')
    shutil.copyfile(mantisshrimp_dir / 'README_MKDOCS.md',
                    dest_dir / 'readme_mkdocs.md')
    shutil.copyfile(mantisshrimp_dir / 'CHANGING-THE-COLORS.md',
                    dest_dir / 'changing-the-colors.md')    

    # Copy images folder from the template folder to the destination folder
    template_images_dir = Path(template_dir)/'images'
    print("Template folder: ", template_images_dir)
    dest_images_dir = Path(dest_dir)/'images'    
    
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


if __name__ == '__main__':
    generate(mantisshrimp_dir / 'docs' / 'sources')
