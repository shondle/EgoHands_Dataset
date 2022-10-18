import scipy.io as sio
import pandas as pd
import numpy as np

## this is assuming location, activity, viewer, and partner are all passed in (all-inclusive)
## just trying to get this right for now to test functionality
def getMetaBy(*args):

    """
        :param location: OFFICE, COURTYARD, LIVINGROOM
        :param activity: CHESS, JENGA, PUZZLE, CARDS
        :param viewer: B, S, T, H
        :param partner: B, S, T, H
        :param main_split: TRAIN, TEST, VALID
        :return: A list of queried video data by input parameters: location, activity, viewer, partner, main_split
    """

    ## assigning from arguments given
    for arg in args:
        if arg == "Location":
            parameterCheck = "Location"
        elif parameterCheck == "Location":
            inLocation = arg
            parameterCheck = 'n/a'
        if arg == "Activity":
            parameterCheck = "Activity"
        elif parameterCheck == "Activity":
            inActivity = arg
            parameterCheck = 'n/a'
        if arg == "Viewer":
            parameterCheck = "Viewer"
        elif parameterCheck == "Viewer":
            inViewer = arg
            parameterCheck = 'n/a'
        if arg == "Partner":
            parameterCheck = "Partner"
        elif parameterCheck == "Partner":
            inPartner = arg
            parameterCheck = 'n/a'
        if arg == "Main_Split":
            parameterCheck = "Main_Split"
        elif parameterCheck == "Main_Split":
            inMain_Split = arg
            parameterCheck = 'n/a'


    # splitting each input variable so we can check multiple conditions
    location = inLocation.split(", ")
    activity = inActivity.split(", ")
    viewer = inViewer.split(", ")
    partner = inPartner.split(", ")

    # loading metadata.mat
    meta_contents = sio.loadmat('./metadata.mat')
    annotations = meta_contents['video'][0]
    annotations_df = pd.DataFrame(annotations, columns=['video_id', 'partner_video_id', 'ego_viewer_id', 'partner_id',
                                                        'location_id', 'activity_id', 'labelled_frames'])


    # data cleaning operation. when we load the np array -> pandas df object,
    # single entries are actually list-type of length 1
    annotations_df['ego_viewer_id'] = annotations_df['ego_viewer_id'].apply(lambda x: x[0])
    queried_activities = annotations_df.loc[annotations_df['ego_viewer_id'].isin(viewer)]

    annotations_df['partner_id'] = annotations_df['partner_id'].apply(lambda x: x[0])
    queried_activities = queried_activities.loc[annotations_df['partner_id'].isin(partner)]

    annotations_df['location_id'] = annotations_df['location_id'].apply(lambda x: x[0])
    queried_activities = queried_activities.loc[annotations_df['location_id'].isin(location)]

    annotations_df['activity_id'] = annotations_df['activity_id'].apply(lambda x: x[0])
    queried_activities = queried_activities.loc[annotations_df['activity_id'].isin(activity)]

    return queried_activities


if __name__ == '__main__':
    # a main method that runs to this file for debugging purposes.
    print('Begin getMetaBy.py program')
    videos = getMetaBy('Location', 'LIVINGROOM, OFFICE', 'Activity', 'JENGA, CARDS', 'Viewer', 'B, S, T, H', 'Partner', 'B, S, T, H')
    print(videos)
    print('End getMetaBy.py program')

