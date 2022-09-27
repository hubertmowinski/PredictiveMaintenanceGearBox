import numpy as np
from tsai.all import *
import logging
import zarr
my_setup(zarr)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_from_file(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)


file_for_train_data = 'train_data'
file_for_train_data_x = file_for_train_data + '_x'
file_for_train_data_y = file_for_train_data + '_y'

logger.info(f"Save lists with train data to files: {file_for_train_data_x}, {file_for_train_data_y}")
X = list_from_file(file_for_train_data_x)
Y = list_from_file(file_for_train_data_y)

path = Path('data')
if not os.path.exists(path): os.makedirs(path)

logger.info("Create numpy array with X and Y")
X_largest = np.array(X)
y_large = np.array(Y)

logger.info("")
X_large = X_largest[:, :, ::100]

logger.info("")
X_large_zarr = zarr.open(path/'X_large.zarr', mode='w', shape=X_large.shape, dtype=X_large.dtype, chunks=(1, -1, -1)) # chunks=(1, -1, -1) == (1, None, None)
X_large_zarr[:] = X_large

X_large_zarr = zarr.open(path/'X_large.zarr', mode='r')
y_large_zarr = zarr.open(path/'y_large.zarr', mode='w', shape=y_large.shape, dtype=y_large.dtype, chunks=False) # y data is small and don't need to be chunked
y_large_zarr[:] = y_large
splits = TimeSplitter()(y_large)
X_large_zarr

zarr_arr = zarr.open(path/'zarr.zarr', mode='w', shape=len(X), dtype=X.dtype)

logger.info("Split data to train and valid")
#Set show plot to True if you want to see split, but this stops main thread
splits = TimeSplitter()(y_large)

tfms = [None, TSClassification()] # TSClassification == Categorize
batch_tfms = TSStandardize()
dls = get_ts_dls(X_large_zarr, y_large_zarr, splits=splits, tfms=tfms, batch_tfms=batch_tfms, inplace=False, bs=1, num_workers=2) # num_workers = 1 *cpus
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