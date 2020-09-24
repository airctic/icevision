# Transforms

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/tfms/)


`Transforms` are used in the following context:

- Resize and pad images to be fed to a given model,

- Augment the number of images in dataset that a given model will be train on. The augmented images are transformed images that will help the model to be trained on more diverse images, and consequently obtain a more robust trained model that will generally perform better than a model trained with non-augmented images,

- All the transforms are lazy transforms meaning they are applied on-the-fly: in other words, we do not create static transformed images which would increase the storage space

**IceVision Transforms Implementation:**

IceVision lays the foundation to easily integrate different augmentation libraries by using adapters. Out-of-the-box, it implements an adapter for the popular [Albumentations](https://albumentations.readthedocs.io/en/latest/) library. Most of the examples and notebooks that we provide showcase how to use our Albumentations transforms.

In addition, IceVision offers the users the option to create their own adapters using the augmentation library of their choice. They can follow a [similar approach](https://github.com/airctic/icevision/tree/master/icevision/tfms/albumentations) to the one we use to create their own augmentation library adapter.

To ease the users' learning curve, we also provide the [aug_tfms](https://airctic.com/albumentations_tfms/#aug_tfms) function that includes some of the most used transforms. The users can also override the default arguments. Other similar transforms pipeline can also be created by the users in order to be applied to their own use-cases.


## Usage

In the following example, we highlight some of the most common usage of transforms. Transforms are used when we create both the train and valid Dataset objects. 

We often apply different transforms for the train and valid Dataset objects. Train transforms are used to augment the original dataset whereas valid transfoms are used to resize an image to fit the size the model expect.

**Example:** [Source](https://airctic.github.io/icevision/examples/training/)
In this example, there are two points to highlight:

- The train_tfms uses the predefined Albumentations transforms to augment the dataset during the train phase. They are applied on-the-fly (lazy transforms) 

- The valid_tfms serves to resize validation images to the size the model expect 

```python
# Defining transforms - using Albumentations transforms out of the box
train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter(
    [*tfms.A.resize_and_pad(size), tfms.A.Normalize()]
)

# Creating both training and validation datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)
```

**Original Image:**
![image](https://airctic.github.io/icevision/images/sample-image.png)

**Transformed Images:**
![image](https://airctic.github.io/icevision/images/sample-image-tfms.png)

!!! info "Note" 
    Notice how different transforms are applied to the original image. All the transformed have the same size despite applying some crop transforms. The size is preserved by adding padding (grey area) 
