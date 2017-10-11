import matplotlib as mpl
import os


#mpl.use('Agg') # Tried several renderers
#os.environ["THEANO_FLAGS"]="optimizer=None,device=cpu" # MPL works when device=cpu
os.environ["THEANO_FLAGS"]="device=cuda0" # MPL fails when device=cuda0
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]="C:/Users/bstri/AppData/Local/conda/conda/envs/py27/Library/plugins/platforms"
import theano # MPL works when not importing theano
import numpy as np
from discrete_skip_gram.plot_util import write_image

img = np.float32(np.eye(28,28))
write_image(img, 'test.png')