# Datasets

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/datasets/)


`Datasets` are designed to simplify both loading and parsing a wide range of computer vision datasets.

**Main Features:**

- Smart Caching: We cache the data so no need to re-download it.

- Lightweight and fast with a transparent and pythonic API.

- Out-of-the-box parsers to convert different datasets into IceVision Data Format.

IceVision provides several ready-to-use datasets that use both standard annotation format such as COCO and VOC as well as custom annotation formats such [WheatParser](https://airctic.github.io/icevision/custom_parser/) used in the [Kaggle Global Wheat Competition](https://www.kaggle.com/c/global-wheat-detection) 


# Usage

Object detection datasets use different annotations formats (COCO, VOC, and custom formats). IceVision offers different options to parse each one of those formats:


## Case 1: COCO, and VOC compatible datasets

### **Option 1: Using icevision predefined VOC parser**
**Example:** Raccoon - dataset using the predefined VOC parser

```python
# Imports
from icevision.all import *

# WARNING: Make sure you have already cloned the raccoon dataset using the command shown here above
# Set images and annotations directories
data_dir = Path("raccoon_dataset")
images_dir = data_dir / "images"
annotations_dir = data_dir / "annotations"

# Define class_map
class_map = ClassMap(["raccoon"])

# Parser: Use icevision predefined VOC parser
parser = parsers.voc(
    annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map
)

# train and validation records
data_splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(data_splitter)
show_records(train_records[:3], ncols=3, class_map=class_map)
```

!!! info "Note" 
    Notice how we use the predifined [parsers.voc()](https://github.com/airctic/icevision/blob/master/icevision/parsers/voc_parser.py) function:
    
    **parser = parsers.voc(
    annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map
    )**


### **Option 2: Creating both data, and parsers files for the VOC or COCO parsers**

**Example:** Fridge Objects - dataset redefining its VOC parser

Please check out the [fridge folder](https://github.com/airctic/icevision/tree/master/icevision/datasets/fridge) for more information on how this dataset is structured.

```python
# Imports
from icevision.all import *

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

!!! info "Note" 
    Notice how we use a new defined [datasets.fridge.parser()](https://github.com/airctic/icevision/blob/master/icevision/datasets/fridge/parsers.py) function:
    
    **parser = datasets.fridge.parser(data_dir, class_map)**


## Case 2: Dataset using a custom parser

**Example:** PETS - a dataset using its custom parser

Please check out the [fridge folder](https://github.com/airctic/icevision/tree/master/icevision/datasets/fridge) for more information on how this dataset is structured.

```python
# Imports
from icevision.all import *

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
    The datasets interface will always have at least the following functions: [load](https://github.com/airctic/icevision/blob/67b89be104be584eac925faa293256beba084408/icevision/datasets/pets/data.py#L54), [class_map](https://github.com/airctic/icevision/blob/67b89be104be584eac925faa293256beba084408/icevision/datasets/pets/data.py#L50), and [parser](https://github.com/airctic/icevision/blob/67b89be104be584eac925faa293256beba084408/icevision/datasets/pets/parsers.py#L8). You might also have noticed the strong similarity between the 2 examples listed here above. Indeed, only the names of the datasets differ, the rest of the code is the same: That highlights how we both simpified and standardized the process of loading and parsing a given dataset.

!!! info "Note 2" 
    If you would like to create your own dataset, we strongly recommend you following the same file structure, and naming found in the different examples such as the [Fridge Objects dataset](https://github.com/airctic/icevision/tree/master/icevision/datasets/fridge), and the [PETS dataset](https://github.com/airctic/icevision/tree/master/icevision/datasets/pets)    

![image](https://airctic.github.io/icevision/images/datasets-folder-structure.png)

# Disclaimer

Inspired from HuggingFace [nlp](https://github.com/huggingface/nlp), icevision_datasets is a utility library that downloads and prepares computer vision datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you are a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/airctic/icevision/issues). Thanks for your contribution to the ML community!

If you are interested in learning more about responsible AI practices, including fairness, please see [Google AI's Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).