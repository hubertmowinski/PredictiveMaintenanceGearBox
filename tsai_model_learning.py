from tsai.all import *
from read_data_and_prepare_to_training import list_from_file
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


file_for_train_data = 'train_data'
file_for_train_data_x = file_for_train_data + '_x'
file_for_train_data_y = file_for_train_data + '_y'

logger.info(f"Save lists with train data to files: {file_for_train_data_x}, {file_for_train_data_y}")
X = list_from_file(file_for_train_data_x)
Y = list_from_file(file_for_train_data_y)

logger.info("Split data to train and valid")
#Set show plot to True if you want to see split, but this stops main thread
splits = get_splits(Y, valid_size=.2, stratify=True, random_state=23, shuffle=True, show_plot=False)

tfms  = [None, TSClassification()] # TSClassification == Categorize
batch_tfms = TSStandardize()
dls = get_ts_dls(X, Y, splits=splits, tfms=tfms, batch_tfms=batch_tfms)
dls.show_batch(sharey=True)

logger.info("Initialize learner")
learn = ts_learner(dls, metrics=accuracy, cbs=ShowGraph())
logger.info("Save learner as stage0")
learn.save('stage0')

logger.info("Initialize learner")
learn.fit_one_cycle(10, lr_max=1e-3)
learn.save('stage1')
learn.recorder.plot_metrics()

PATH = Path('./Multiclass.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')