# About Datasets

`Datasets` is a lightweight and easy to use library to share common Computer Vision datasets.

Main Features: -
- Smart Caching: We cache the data so no need to re-download.
- Lightweight and fast with a transparent and pythonic API.
- We provide Parsers for converting these datasets into Mantisshrimp Data Format.

It is currently under development. As of now we provide COCO, VOC and Pets datasets.

# Usage

Using `datasets` is made to be very simple to use.

Here is a quick example

```
from mantisshrimp import datasets

# Download the pets datasets. It will be saved in a folder called .mantisshrimp
data_dir = datasets.pets.load()

# Create a Parser for dataset.
parser = datasets.pets.parser(data_dir)

# Create a trian and validation dataset
data_splitter = RandomSplitter([.8, .2])

# Parse the data into train and validation
train_records, valid_records = parser.parse(data_splitter)

# List of all available classes
pet_classes = datasets.pets.CLASSES

```
Note: - The datasets interface will always have at least two functions: `load` and `parser` and an attribute `CLASSES`


# Disclaimer

Inspired from HuggingFace [nlp](https://github.com/huggingface/nlp) mantisshrimp_datasets is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/airctic/mantisshrimp/issues). Thanks for your contribution to the ML community!

If you're interested in learning more about responsible AI practices, including fairness, please see [Google AI's Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/).