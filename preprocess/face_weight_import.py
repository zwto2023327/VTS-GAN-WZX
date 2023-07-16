import SPIGA.colab_tutorials.video_tools.record as vid_util

webcam_video_path = '/content/test.mp4'
vid_util.record_video(webcam_video_path)
import os
from SPIGA.spiga.demo.app import video_app

# MP4 input path: Webcam recorded video or uploaded one.
# video_path = '/content/<path_to_your_video>'
video_path = webcam_video_path
output_path= '/content/output'  # Processed video storage

# Process video
video_app(video_path,
          spiga_dataset='wflw',               # Choices=['wflw', '300wpublic', '300wprivate', 'merlrav']
          tracker='RetinaSort',               # Choices=['RetinaSort', 'RetinaSort_Res50']
          save=True,
          output_path=output_path,
          visualize=False,
          plot=['fps', 'face_id', 'landmarks', 'headpose'])


# Convert Opencv video to Colab readable format
video_name = video_path.split('/')[-1]
video_output_path = os.path.join(output_path, video_name)
video_colab_path = os.path.join(output_path, video_name[:-4]+'_colab.mp4')
!ffmpeg -i '{video_output_path}' '{video_colab_path}'