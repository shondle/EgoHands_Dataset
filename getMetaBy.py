import scipy.io as sio
import numpy as np

## this is assuming location, activity, viewer, and partner are all passed in (all-inclusive)
## just trying to get this right for now to test functionality
def getMetaBy(Location, Activity, Viewer, Partner):

    # splitting each input variable so we can check multiple conditions
    location = Location.split(", ")
    activity = Activity.split(", ")
    viewer = Viewer.split(", ")
    partner = Partner.split(", ")

    # loading metadata.mat
    meta_contents = sio.loadmat('./metadata.mat')
    struct = meta_contents['video']

    # creating an empty numpy array to append to later
    videos = np.empty(0)

    # looping through to find which combinations are found in the dataset, and returning the video
    # (when all four inputs given)
    # sorry for the weird alphabet structure in the for loops, I'll fix that
    for a in range(len(location)):
        for b in range(len(activity)):
            for d in range(len(viewer)):
                for e in range(len(partner)):
                    for c in range(48):
                        matrix = struct[0][c]
                        if ((matrix[4] == location[a]) & (matrix[5] == activity[b]) & (matrix[2] == viewer[d]) & (matrix[3] == partner[e])).any():
                            # print(c) returns the correct index for the video in the list with the inputs given
                            # so, I know that the for loops are built correctly, at least conceptually
                            print(c)

                            # however I can't figure out how to return and store this correctly
                            # this attempts to append labelled frames struct for identified video
                            videos = np.append(videos, matrix[6])

                            # this attempts to append all the video's data given in metadata.mat
                            #videos = np.append(videos, struct[0][c])
    return videos

    # when calling I get the following error message -

    # The DType <class 'numpy.dtype[float64]'> could not be promoted by <class 'numpy.dtype[void]'>.
    # This means that no common DType exists for the given inputs.
    # For example they cannot be stored in a single array unless the dtype is `object`.
    # The full list of DTypes is: (<class 'numpy.dtype[float64]'>, <class 'numpy.dtype[void]'>)

# Perhaps this is relevant?
# https://stackoverflow.com/questions/55437498/numpy-append-typeerror-invalid-type-promotion

# Also, do I need to return the entire struct set, or just the labelled frames?
