# Recognize
Python based UI for choosing images and using ML algorithms (FRCNN, SSD Inception, SSD MobileNet) to label them. Various options are also provided.

![Recognize UI](https://github.com/SurajViitk/recognize/blob/master/screenshot/recognize.jpg)

Sample ScreenRecording-
https://drive.google.com/open?id=1NnT5_huRoxYR65tYrf49l9cvkUHmY-SQ

To use this project, just 
1) Download and load models from the links and copy in actual_model folder
    Links - i) http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
            ii) http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
            iii) http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
    Example directory structure is /actual_model/faster_rcnn_inception_v2_coco_2018_01_28/ containing all the files
2) run Recognize.py (in this implementation, model gets load after clicking Detect) or Recognize2.py (here, models are preloaded, high on RAM but faster) using python3. 
