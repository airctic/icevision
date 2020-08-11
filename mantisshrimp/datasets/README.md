# Datasets

`Datasets` are designed to simplify both loading and parsing a wide range of computer vision datasets.

**Main Features:**

- Smart Caching: We cache the data so no need to re-download it.

- Lightweight and fast with a transparent and pythonic API.

- Ready-to-use standard parsers (COCO, and VOC) as well as some custom parsers to convert datasets into Mantisshrimp Data Format.

Mantisshrimp provides several ready-to-use datasets that use both standard annotation format such as COCO and VOC as well as custom annotation formats such [WheatParser](https://airctic.github.io/mantisshrimp/custom_parser/) used in the [Kaggle Global Wheat Competition](https://www.kaggle.com/c/global-wheat-detection) 


# Usage

Here are some examples of datasets with their corresponding parsers:

## Fridge Objects: a Dataset using the VOC parser
Please check out the [fridge folder](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/datasets/fridge) for more information on how this dataset is structured.

```python
# Imports
from mantisshrimp.all import *

# Load the Fridge Objects dataset
data_dir = datasets.fridge.load()

# Get the class_map, a utility that maps from number IDs to classs names
class_map = datasets.fridge.class_map()

# Randomly split our data into train/valid
data_splitter = RandomSplitter([0.8, 0.2])

# VOC parser: provided out-of-the-box
parser = datasets.fridge.parser(data_dir, class_map)
train_records, valid_records = parser.parse(data_splitter)

# shows images with corresponding labels and boxes
show_records(train_records[:3], ncols=3, class_map=class_map)
```

## PETS: a Dataset using a custom parser
Please check out the [fridge folder](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/datasets/fridge) for more information on how this dataset is structured.

```python
# Imports
from mantisshrimp.all import *

# Load the PETS dataset
path = datasets.pets.load()

# Get the class_map, a utility that maps from number IDs to classs names
class_map = datasets.pets.class_map()

# Randomly split our data into train/valid
data_splitter = RandomSplitter([0.8, 0.2])

# PETS parser: provided out-of-the-box
parser = datasets.pets.parser(data_dir=path, class_map=class_map)
train_records, valid_records = parser.parse(data_splitter)

# shows images with corresponding labels and boxes
show_records(train_records[:6], ncols=3, class_map=class_map, show=True)

```

!!! info "Note 1" 
    The datasets interface will always have at least the following functions: `load`, `parser`, and `class_map`. You might also have noticed the stong similarity between the 2 examples listed here above. Indeed, only the names of the datasets differ, the rest of the code is the same: That highlights how we both simpified and standardized the process of loading and parsing a given dataset.

!!! info "Note 2" 
    If you would like to create your own dataset, we strongly recommend you following the same file structure, and naming found in the different examples such as the [Fridge Objects dataset](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/datasets/fridge), and the [PETS dataset](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/datasets/pets)    

![image](https://airctic.github.io/mantisshrimp/images/datasets-folder-structure.png)

# Disclaimer

Inspired from HuggingFace [nlp](https://github.com/huggingface/nlp), mantisshrimp_datasets is a utility library that downloads and prepares computer vision datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you are a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/airctic/mantisshrimp/issues). Thanks for your contribution to the ML community!

If you are interested in learning more about responsible AI practices, including fairness, please see [Google AI's Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).