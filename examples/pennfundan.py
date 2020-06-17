from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *

source = get_pennfundan_data()
parser = PennFundanParser(source)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)
