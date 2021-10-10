import pytest
from icevision.all import *

# test - full 0% overlap
@pytest.fixture()
def setup_pred_no_overlap():
    # synthetic data to test
    gt = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])


    pred = np.asarray([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
    ])

    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction

# testing metric 
@pytest.fixture()
def expected_binary_dice_output_no_overlap():

    return {'dummy_value_for_fastai': 0.0}

# @pytest.fixture()
def test_binary_dice_no_overlap(setup_pred_no_overlap, expected_binary_dice_output_no_overlap):

    pred = setup_pred_no_overlap

    dice = BinaryDiceCoefficient()
    dice.accumulate([pred])

    assert dice.finalize() == expected_binary_dice_output_no_overlap



# test - full 100% overlap
@pytest.fixture()
def setup_pred_full_overlap():
    # synthetic data to test
    gt = np.asarray([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
    ])

    pred = np.asarray([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
    ])


    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction

# testing metric 
@pytest.fixture()
def expected_binary_dice_output_full_overlap():

    return {'dummy_value_for_fastai': 1.0}

# @pytest.fixture()
def test_binary_dice_full_overlap(setup_pred_full_overlap, expected_binary_dice_output_full_overlap):

    pred = setup_pred_full_overlap

    dice = BinaryDiceCoefficient()
    dice.accumulate([pred])

    assert dice.finalize() == expected_binary_dice_output_full_overlap


# test - 25% overlap
@pytest.fixture()
def setup_pred_quarter_overlap():
    # synthetic data to test
    gt = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    pred = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction

# testing metric 
@pytest.fixture()
def expected_binary_dice_output_quarter():

    return {'dummy_value_for_fastai': 0.4}

# @pytest.fixture()
def test_binary_dice_quarter_overlap(setup_pred_quarter_overlap, expected_binary_dice_output_quarter):

    pred = setup_pred_quarter_overlap

    dice = BinaryDiceCoefficient()
    dice.accumulate([pred])

    assert dice.finalize() == expected_binary_dice_output_quarter


# test - 50% overlap
@pytest.fixture()
def setup_pred_half_overlap():
    # synthetic data to test
    gt = np.asarray([
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,0,0],
    ])

    pred = np.asarray([
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1],
    ])

    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction

# testing metric 
@pytest.fixture()
def expected_binary_dice_output_half_overlap():

    return {'dummy_value_for_fastai': 0.6666666666666666}

# @pytest.fixture()
def test_binary_dice_half_overlap(setup_pred_half_overlap, expected_binary_dice_output_half_overlap):

    pred = setup_pred_half_overlap

    dice = BinaryDiceCoefficient()
    dice.accumulate([pred])

    assert dice.finalize() == expected_binary_dice_output_half_overlap



# test - patterened overlap
@pytest.fixture()
def setup_pred_patterned_overlap():
    # synthetic data to test
    gt = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    pred = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction


# testing metric 
@pytest.fixture()
def expected_binary_dice_output_patterned_overlap():

    return {'dummy_value_for_fastai': 0.3333333333333333}


# @pytest.fixture()
def test_binary_dice_patterned_overlap(setup_pred_patterned_overlap, expected_binary_dice_output_patterned_overlap):

    pred = setup_pred_patterned_overlap

    dice = BinaryDiceCoefficient()
    dice.accumulate([pred])

    assert dice.finalize() == expected_binary_dice_output_patterned_overlap


