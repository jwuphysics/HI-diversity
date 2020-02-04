from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

from ranger import Ranger
from tqdm.notebook import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

PATH = Path('..').resolve()

sys.path.append(f'{PATH}/src')
from mxresnet import *


flat_np = lambda arr: to_np(arr.reshape(-1))

def get_xgass_data(bs=32, sz=224, legacy=False, seed=None):
    df = pd.read_csv(f'{PATH}/data/xGASS_representative_sample.csv')

    if legacy:
        image_stats = [tensor([-0.0443, -0.0643, -0.0726]), tensor([0.9129, 0.8815, 0.8759])]
    else:
        image_stats = [tensor([-0.0169, -0.0105, -0.0004]), tensor([0.9912, 0.9968, 1.0224])]

    tfms = get_transforms(
        do_flip=True,
        flip_vert=True,
        max_zoom=1.0,
        max_rotate=15.0,
        max_lighting=0,
        max_warp=0,
    )

    bs = bs
    sz = sz

    src = (
        ImageList.from_df(
            df, 
            path=PATH, 
            folder='images-xGASS' + ('-legacy' if legacy else ''), 
            suffix='.jpg', 
            cols='GASS')
                .split_by_rand_pct(0.2, seed=seed)
                .label_from_df(cols=['lgGF'])
    )

    data = (
        src.transform(tfms, size=sz)
           .databunch(bs=bs)
           .normalize(image_stats)
    )

    return data

def get_combined_data(bs=32, sz=224, legacy=False, seed=None):
    df = pd.read_csv(PATH/'data'/'combined.csv', index_col=0)

    if legacy:
        image_stats = [tensor([-0.0443, -0.0643, -0.0726]), tensor([0.9129, 0.8815, 0.8759])]
    else:
        image_stats = [tensor([-0.0169, -0.0105, -0.0004]), tensor([0.9912, 0.9968, 1.0224])]
        
    tfms = get_transforms(
        do_flip=True,
        flip_vert=True,
        max_zoom=1.0,
        max_rotate=15.0,
        max_lighting=0,
        max_warp=0,
    )

    bs = bs
    sz = sz

    src = (
        ImageList.from_df(
            df, 
            path=PATH, 
            folder='images-xGASS' + ('-legacy' if legacy else ''), 
            suffix='.jpg', 
            cols='GASS')
                .split_by_rand_pct(0.2, seed=seed)
                .label_from_df(cols='logfgas',  label_cls=FloatList)
    )

    data = (src.transform(tfms, size=sz)
               .databunch(bs=bs)
               .normalize(image_stats)
    )

    return data


def custom_cnn_learner(
    bs=32, 
    sz=224, 
    legacy=False, 
    cnn_model=mxresnet18, 
    load_model=None
):

    data = get_combined_data(bs=bs, sz=sz, legacy=legacy)

    model = cnn_model()
    model[-1] = nn.Linear(model[-1].in_features, 1, bias=True).cuda()

    learn = Learner(
        data,
        model=model,
        opt_func=partial(Ranger),
        loss_func=root_mean_squared_error,
        wd=1e-4,
        bn_wd=False,
        true_wd=True,
    )

    return learn

if __name__ == '__main__':
    
    learn = custom_cnn_learner(bs=32, sz=224, legacy=False)

