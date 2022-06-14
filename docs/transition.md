# How to transition

## Parser
The major modification was in the `Parser` API, the concept of a `ParserMixin` was completely removed. Now, instead of starting by defining the parser, we start by defining the record:

**Before:**
```python
class MyParser(
    parsers.Parser,
    parsers.FilepathMixin,
    parsers.LabelsMixin,
    parsers.BBoxesMixins,
):
```

**After:**
```python
template_record = BaseRecord(
    (
        FilepathRecordComponent(),
        InstancesLabelsRecordComponent(),
        BBoxesRecordComponent(),
    )
)
```

We then continue to generate the parser template:

**Before:**
```python
MyParser.generate_template()
```

**After:**
```python
Parser.generate_template(template_record)
```

And here's how the parser class looks:


**Before:**
```python
class ChessParser(
    parsers.Parser,
    parsers.FilepathMixin,
    parsers.LabelsMixin,
    parsers.BBoxesMixins,
):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = pd.read_csv(data_dir / "annotations.csv")

        class_map = ClassMap(list(self.df['label'].unique()))
        super().__init__(class_map=class_map)
        
    def __iter__(self) -> Any:
        yield from self.df.itertuples()
        
    def __len__(self) -> int:
        return len(self.df)
        
    def imageid(self, o) -> Hashable:
        return o.filename

    def filepath(self, o) -> Union[str, Path]:
        return self.data_dir / 'images' / o.filename

    def image_width_height(self, o) -> Tuple[int, int]:
        return o.width, o.height

    def labels(self, o) -> List[int]:
        return [o.label]

    def bboxes(self, o) -> List[BBox]:
        return [BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)]
```

**After:**
```python
class ChessParser(Parser):
    def __init__(self, template_record, data_dir):
        super().__init__(template_record=template_record)
        
        self.data_dir = data_dir
        self.df = pd.read_csv(data_dir / "annotations.csv")
        self.class_map = ClassMap(list(self.df['label'].unique()))
        
    def __iter__(self) -> Any:
        yield from self.df.itertuples()
        
    def __len__(self) -> int:
        return len(self.df)
        
    def imageid(self, o) -> Hashable:
        return o.filename
        
    def parse_fields(self, o, record):
        record.set_filepath(self.data_dir / 'images' / o.filename)
        record.set_img_size(ImgSize(width=o.width, height=o.height))
        
        record.detect.set_class_map(self.class_map)
        record.detect.add_bboxes([BBox.from_xyxy(o.xmin, o.ymin, o.xmax, o.ymax)])
        record.detect.add_labels([o.label])
```

## Record

The attributes on the record are now separated by task, to access `bboxes` you now have to do `record.detection.bboxes` instead of `record.bboxes`. Common attributes (not specific to one task) can still be accessed directly: `record.filepath`.

You can use `print` to check the attributes on the record and how to access them:

```python
BaseRecord

common: 
	- Image size ImgSize(width=416, height=416)
	- Image ID: 3
	- Filepath: /home/lgvaz/.icevision/data/chess_sample/chess_sample-master/images/e79deba8fe520409790b601ad61da4ee_jpg.rf.016bc04dee292f80d1f975931f32bc21.jpg
	- Image: None
detection: 
	- BBoxes: [<BBox (xmin:208, ymin:88, xmax:226, ymax:128)>]
	- Labels: [5]
```

All fields under `common` can be accessed directly, the others have to be
specified by it's corresponding task.
