# enter matlab.engine.shareEngine in matlab before running

import matlab.engine
import numpy as np
import scipy.io as sio
from getMetaBy import getMetaBy

# constructing matlab engine to reference
eng = matlab.engine.start_matlab()

# direct engine to path with matlab files and dataset located
eng.addpath(r'')

# supplying inputs for Location, Activity, Viewer, and Partner into getMetaBy
# don't need to run the MATLAB Engine for getMetaBy
videos1 = getMetaBy('COURTYARD', 'CARDS', 'B, H', 'T, S')
