import os

#  getFramePath(vid, i) returns the file path to the "i"th labeled frame in video "vid", where "vid"
#  is an EgoHands video metadata structure.
#
#
#    For full dataset details, see the <a href="matlab: web('http://vision.soic.indiana.edu/egohands')">EgoHands project website</a>.
#
#    See also getBoundingBoxes, getMetaBy, getSegmentationMask, showLabelsOnFrame

def getFramePath(video, i):
    base_path = os.path.join(os.getcwd(), '_LABELLED_SAMPLES', (video.loc['video_id'])[0])
    frame_path = os.path.join(base_path, 'frame_%(number)04d.jpg' % {'number': (video.loc['labelled_frames'])[0][i][0][0][0]})
    return frame_path
