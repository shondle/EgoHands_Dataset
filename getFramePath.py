from pathlib import Path

def getFramePath(video, i):
    # change path to generic before sending
    base_path = Path(r'.\egohands_data\_LABELLED_SAMPLES'
                       + '/'
                       + (video.loc['video_id'])[0])
    frame_path = base_path.joinpath(Path('frame_%(number)04d.jpg' % \
                                        {"number": (video.loc['labelled_frames'])[0][i][0][0][0]}))
    return frame_path
