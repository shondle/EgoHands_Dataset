# enter matlab.engine.shareEngine in matlab before running

import matlab.engine
import numpy as np

## constructing matlab engine to reference
eng = matlab.engine.start_matlab()

## direct engine to path with matlab files and dataset located
eng.addpath(r'')

# structure arrays not supported in MATLAB Engine API, so this must be done manually
# videos = {}
# videos = eng.getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
