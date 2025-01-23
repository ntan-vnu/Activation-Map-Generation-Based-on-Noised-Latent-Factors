IMAGE_SIZE = (256, 256, 3)
NUM_CLASS = 8
DATA_DIR = 'datasets/BreakHis_v1'
CHECKPOINT_DIR = 'checkpoints/'
CLASS_NAMES = ['adenosis', 'fibroadenoma',
               'phyllodes_tumor', 'tubular_adenoma',
               'ductal_carcinoma', 'lobular_carcinoma',
               'mucinous_carcinoma', 'papillary_carcinoma']
CGUNET_CHECKPOINT = CHECKPOINT_DIR + '/CGUNet-20240813233300'
CLASSIFIER_CHECKPOINT = CHECKPOINT_DIR + '/classifier256-20240815102200'
GENERATOR_CHECKPOINT = CHECKPOINT_DIR+'/generator-CGUNet-20240813233300'