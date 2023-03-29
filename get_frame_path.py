"""
get_frame_path(vid, i) returns the file path to the "i"th labeled frame in video "vid", where "vid"
is an EgoHands video metadata structure.
For full dataset details, see the
<a href="matlab: web('http://vision.soic.indiana.edu/egohands')">EgoHands project website</a>.
See also get_bounding_boxes, get_meta_by, get_segmentation_mask, showLabelsOnFrame
"""

import os

def get_frame_path(video, i):
    """Creates path starting from _LABBELLED_SAMPLE_. If this is resulting in an error,
    check again to make sure that the _LABELLED_SAMPLE_ folder is in the same directory
    as the rest of the code"""

    base_path = os.path.join(os.getcwd(), './EgoHands_Dataset/_LABELLED_SAMPLES', (video.loc['video_id'])[0])
    frame_path = os.path.join(base_path, 'frame_%(number)04d.jpg' %
                              {'number': (video.loc['labelled_frames'])[0][i][0][0][0]})
    return frame_path
