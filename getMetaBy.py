import scipy.io as sio
import pandas as pd
import numpy as np

# From the original EgoHands code from Indiana University-
# getMetaBy  Returns EgoHands video metadata structures for videos which
# match the argument filters. For a description of the metadata structure
# see buildMetadata help.
#
#   C = getMetaBy() returns metadata for all videos.
#
#   C = getMetaBy(FilterName, Value, ...) returns metadat for all videos
#   matching the filters. Possible filters and values are listed below:
#
#   Filter            Possible Values        Info
#   ----------        --------------------   ------------------------
#   'Location'        'OFFICE','COURTYARD',  Video background location
#                     'LIVINGROOM'
#
#   'Activity'        'CHESS','JENGA',       Activity in video
#                     'PUZZLE','CARDS'
#
#   'Viewer'          'B','S','T','H'        Identity of egocentric viewer
#
#   'Partner'         'B','S','T','H'        Identity of egocentric partner
#
#   Multiple filters and values can be mixed, for example:
#   getMetaBy('Location','OFFICE, COURTYARD', 'Activity','CHESS', 'Viewer', 'B,S,T')
#   would return all videos of Chess played with B,S, or T as the egocentric
#   observer filmed at the Office or Courtyard locations.

def getMetaBy(*args):

    """
        :param location: OFFICE, COURTYARD, LIVINGROOM
        :param activity: CHESS, JENGA, PUZZLE, CARDS
        :param viewer: B, S, T, H
        :param partner: B, S, T, H
        :param main_split: TRAIN, TEST, VALID
        :return: A list of queried video data by input parameters: location, activity, viewer, partner, main_split
    """

    inLocation = 'OFFICE, COURTYARD, LIVINGROOM'
    inViewer = 'B, S, T, H'
    inPartner = 'B, S, T, H'
    inActivity = 'CHESS, JENGA, PUZZLE, CARDS'


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
    a = 5
    # print('Begin getMetaBy.py program')
    # videos = getMetaBy('Location', 'LIVINGROOM, OFFICE', 'Activity', 'JENGA, CARDS', 'Viewer', 'B, S, T, H', 'Partner', 'B, S, T, H')
    videos = getMetaBy('Location', 'COURTYARD', 'Activity', 'PUZZLE')
    ## first one is PUZZLE_COURTYARD_B_S
    print((videos.iloc[0]).loc['video_id'])
    # print(((videos.iloc[0]).loc['labelled_frames'])[0][7][0][0][0])
    # print('End getMetaBy.py program')

