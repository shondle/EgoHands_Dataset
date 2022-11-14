"""
From the original EgoHands code from Indiana University-
get_meta_by  Returns EgoHands video metadata structures for videos which
match the argument filters. For a description of the metadata structure
see buildMetadata help.

  C = get_meta_by() returns metadata for all videos.

  C = get_meta_by(FilterName, Value, ...) returns metadat for all videos
  matching the filters. Possible filters and values are listed below:

  Filter            Possible Values        Info
  ----------        --------------------   ------------------------
  'Location'        'OFFICE','COURTYARD',  Video background location
                    'LIVINGROOM'

  'Activity'        'CHESS','JENGA',       Activity in video
                    'PUZZLE','CARDS'

  'Viewer'          'B','S','T','H'        Identity of egocentric viewer

  'Partner'         'B','S','T','H'        Identity of egocentric partner

  Multiple filters and values can be mixed, for example:
  get_meta_by('Location','OFFICE, COURTYARD', 'Activity','CHESS', 'Viewer', 'B,S,T')
  would return all videos of Chess played with B,S, or T as the egocentric
  observer filmed at the Office or Courtyard locations.
"""
import scipy.io as sio
import pandas as pd

def get_meta_by(*args):

    """
        :param location: OFFICE, COURTYARD, LIVINGROOM
        :param activity: CHESS, JENGA, PUZZLE, CARDS
        :param viewer: B, S, T, H
        :param partner: B, S, T, H
        :param main_split: TRAIN, TEST, VALID
        :return: A list of queried video data by input parameters:
                location, activity, viewer, partner, main_split
    """

    location_params = 'OFFICE, COURTYARD, LIVINGROOM'
    viewer_params = 'B, S, T, H'
    partner_params = 'B, S, T, H'
    activity_params = 'CHESS, JENGA, PUZZLE, CARDS'


    ## assigning from arguments given
    for arg in args:
        if arg == "Location":
            parameter_check = "Location"
        elif parameter_check == "Location":
            location_params = arg
            parameter_check = 'n/a'
        if arg == "Activity":
            parameter_check = "Activity"
        elif parameter_check == "Activity":
            activity_params = arg
            parameter_check = 'n/a'
        if arg == "Viewer":
            parameter_check = "Viewer"
        elif parameter_check == "Viewer":
            viewer_params = arg
            parameter_check = 'n/a'
        if arg == "Partner":
            parameter_check = "Partner"
        elif parameter_check == "Partner":
            partner_params = arg
            parameter_check = 'n/a'


    # splitting each input variable so we can check multiple conditions

    location = location_params.split(", ")
    activity = activity_params.split(", ")
    viewer = viewer_params.split(", ")
    partner = partner_params.split(", ")

    # loading metadata.mat
    meta_contents = sio.loadmat('./metadata.mat')
    annotations = meta_contents['video'][0]
    annotations_df = pd.DataFrame(annotations, columns=['video_id', 'partner_video_id',
                                                        'ego_viewer_id', 'partner_id',
                                                        'location_id', 'activity_id',
                                                        'labelled_frames'])


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
