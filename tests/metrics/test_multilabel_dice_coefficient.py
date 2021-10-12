import pytest
from icevision.all import *


@pytest.fixture()
def setup_cases():
    return [
        {
            'name': "no_overlap",
            'pred_mask': np.asarray([
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
                ]),
            'gt_mask': np.asarray([
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
                ]),
            'expected_value': {'dummy_value_for_fastai': 0.0},
            'binary':  True
        },

        {
            'name': 'full_overlap',
            'pred_mask': np.asarray([
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
            ]),
            'gt_mask': np.asarray([
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
            ]),
            'expected_value': {'dummy_value_for_fastai': 1.0},
            'binary': True
        },
        
        {
            
            'name': 'quarter_overlap',
            'pred_mask': np.asarray([
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
                [0,0,0,0,0,0,0,0,0,0,2,2],
                [0,0,0,0,0,0,0,0,0,0,2,2],
            ]),
            'gt_mask': np.asarray([
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,2,2,2,2],
                [0,0,0,0,0,0,0,0,2,2,2,2],
                [0,0,0,0,0,0,0,0,2,2,2,2],
                [0,0,0,0,0,0,0,0,2,2,2,2],
            ]),
            'expected_value': {'dummy_value_for_fastai': 0.6516129032258065},
            'binary': False
        },

        {
            'name': 'half_overlap',
            'pred_mask': np.asarray([
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
                [1,1,1,0,0,0,0,0,0,2,2,2],
            ]),
            'gt_mask': np.asarray([
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,1,2,2,2,2,2,2],
            ]),
            'expected_value': {'dummy_value_for_fastai': 0.3333333333333333},
            'binary': False
        },

        {
            'name': 'patterned_overlap',
            'pred_mask': np.asarray([
                [2,2,2,2,0,0,0,0,0,0,0,0],
                [2,2,0,0,1,1,0,0,0,0,0,0],
                [2,2,0,0,1,1,0,0,0,0,0,0],
                [0,0,2,2,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,1,0,0,0,0],
                [2,2,2,2,0,0,1,1,0,0,0,0],
                [2,2,2,2,0,0,0,0,0,0,0,0],
                [2,2,2,2,0,0,0,0,0,0,0,0],
            ]),
            'gt_mask': np.asarray([
                [2,2,2,2,0,0,0,0,0,0,0,0],
                [2,2,2,2,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [2,2,0,0,1,1,1,1,0,0,0,0],
                [2,2,0,0,1,1,1,1,0,0,0,0],
                [2,2,2,2,0,0,0,0,0,0,0,0],
            ]),
            'expected_value': {'dummy_value_for_fastai': 0.5872274143302181},
            'binary': False
        },

            {
            'name': 'test_matrix',
            'pred_mask': np.asarray([
                [0,0,1,1],
                [0,0,1,1],
                [2,2,1,1],
            ]),
            'gt_mask': np.asarray([
                [0,0,0,0],
                [1,1,1,1],
                [2,2,2,2],
            ]),
            'expected_value': {'dummy_value_for_fastai': 0.45},
            'binary': False
        }

        ]


# @pytest.fixture()
def test_multilabel_dice_metric(setup_cases):

    cases = setup_cases

    for case in cases:

        # setup pred record
        pred_record = BaseRecord((
            ImageRecordComponent(),
            SemanticMaskRecordComponent(),
            ClassMapRecordComponent(task=tasks.segmentation),))

        pred_record.segmentation.set_class_map(ClassMap(["square"]))
        pred_record.segmentation.set_mask_array(MaskArray(case['pred_mask']))

        # setup ground truth record
        gt_record = BaseRecord((
            ImageRecordComponent(),
            SemanticMaskRecordComponent(),
            ClassMapRecordComponent(task=tasks.segmentation),)) 

        gt_record.segmentation.set_class_map(ClassMap(["square"]))
        gt_record.segmentation.set_mask_array(MaskArray(case['gt_mask']))

        # w, h = imgA.shape[0], imgA.shape[1]
        w, h = case['gt_mask'].shape[0], case['gt_mask'].shape[1]

        gt_record.set_img_size(ImgSize(w,h), original=True)

        prediction = Prediction(pred=pred_record, ground_truth=gt_record)


        # print(f'Results for test with {name}: {results}')

        if case['binary'] == True:

            multi_dice = MulticlassDiceCoefficient(classes_to_exclude=["background"])
            multi_dice.accumulate([prediction])
            results = multi_dice.finalize()
        
        else:

            multi_dice = MulticlassDiceCoefficient()
            multi_dice.accumulate([prediction])
            results = multi_dice.finalize()

        name = case['name']
        
        # return f'Results for test with {name}: {results}'

        assert results == case['expected_value']
