from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import os
import pathlib

import tensorflow as tf
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

__appname__ = "RECOGNIZE"

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self):
      super(MainWindow, self).__init__()

      #image settings
      self.currIndex = 0
      self.maxIndex = 0
      self.threshold = 0.7
      self.req_classes = []


      #view
      self.setWindowTitle(__appname__)
      self.horizontalLayout = QHBoxLayout()
      self.createLeftBox()
      self.createMidBox()
      self.createRightBox()
      self.horizontalLayout.addLayout(self.verticalLeftLayout,1)
      self.horizontalLayout.addWidget(self.midLabel,3)
      self.horizontalLayout.addLayout(self.verticalRightLayout,1)
      # self.setLayout(self.horizontalLayout)
      wid = QWidget(self)
      self.setCentralWidget(wid)

      self.resize(1000, 600) 
      wid.setLayout(self.horizontalLayout)
      self.show()
      self.PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'



      #controller




    def createLeftBox(self):
      self.verticalLeftLayout= QVBoxLayout()
      openFolderButton = QPushButton("Open Folder")
      openFolderButton.setFixedWidth(140)
      openFolderButton.clicked.connect(self.openFolder)
      nextImageButton = QPushButton("Next Image")
      nextImageButton.setFixedWidth(140)
      nextImageButton.clicked.connect(self.nextImage)
      prevImageButton = QPushButton("Previous Image")
      prevImageButton.setFixedWidth(140)
      prevImageButton.clicked.connect(self.prevImage)
      saveAnnotationButton = QPushButton("Save Annotation")
      saveAnnotationButton.setFixedWidth(140)
      self.verticalLeftLayout.addWidget(openFolderButton)
      self.verticalLeftLayout.addWidget(nextImageButton)
      self.verticalLeftLayout.addWidget(prevImageButton)
      self.verticalLeftLayout.addWidget(saveAnnotationButton)

    def createMidBox(self):
      self.midLabel = QLabel("Image")
      pixmap = QPixmap()
      # pixmap.setSize(800, 800)
      self.midLabel.setPixmap(pixmap)

    def createRightBox(self):
      self.verticalRightLayout = QVBoxLayout()
      
      modelVLayout = QVBoxLayout()
      modelLabel = QLabel("Select Model")
      radioButton1 = QRadioButton("FRCNN")
      radioButton1.algo_dir = "actual_model/faster_rcnn_inception_v2_coco_2018_01_28"
      radioButton1.toggled.connect(self.radioListner)
      radioButton2 = QRadioButton("SSD Inception")
      radioButton2.algo_dir = "actual_model/ssd_inception_v2_coco_2018_01_28"
      radioButton2.toggled.connect(self.radioListner)
      radioButton3 = QRadioButton("SSD MobileNet")
      radioButton3.algo_dir = "actual_model/ssd_mobilenet_v1_coco_2018_01_28"
      radioButton3.toggled.connect(self.radioListner)
      radioButton1.setChecked(True)
      modelVLayout.addWidget(modelLabel)
      modelVLayout.addWidget(radioButton1)
      modelVLayout.addWidget(radioButton2)
      modelVLayout.addWidget(radioButton3)

      thresholdVLayout = QVBoxLayout()
      thresholdLabel = QLabel("Detection Threshold")
      spinBox = QDoubleSpinBox()
      spinBox.setDecimals(2)
      spinBox.setValue(0.7)
      spinBox.setSingleStep(0.05)
      spinBox.setMinimum(0.0)
      spinBox.setMaximum(1.0)
      spinBox.valueChanged.connect(self.spinBoxListener)
      thresholdVLayout.addWidget(thresholdLabel)
      thresholdVLayout.addWidget(spinBox)

      labelFilterVLayout = QVBoxLayout()
      labelFilterLabel = QLabel("Label Filter")
      b1 = QCheckBox("Person")
      b1.stateChanged.connect(self.checkBoxListner)
      b1.label_class = 1
      b2 = QCheckBox("Cat")
      b2.label_class = 17
      b2.stateChanged.connect(self.checkBoxListner)
      b3 = QCheckBox("Dog")
      b3.label_class = 18
      b3.stateChanged.connect(self.checkBoxListner)
      b4 = QCheckBox("Bottle")
      b4.label_class = 44
      b4.stateChanged.connect(self.checkBoxListner)
      b5 = QCheckBox("Chair")
      b5.label_class = 62
      b5.stateChanged.connect(self.checkBoxListner)
      labelFilterVLayout.addWidget(labelFilterLabel)
      labelFilterVLayout.addWidget(b1)
      labelFilterVLayout.addWidget(b2)
      labelFilterVLayout.addWidget(b3)
      labelFilterVLayout.addWidget(b4)
      labelFilterVLayout.addWidget(b5)

      detectButton = QPushButton("Detect")
      detectButton.clicked.connect(self.detectButtonclickListner)

      self.verticalRightLayout.addLayout(modelVLayout)
      self.verticalRightLayout.addLayout(thresholdVLayout)
      self.verticalRightLayout.addLayout(labelFilterVLayout)
      self.verticalRightLayout.addWidget(detectButton)

    def openFolder(self):
      self.currFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
      self.imagePaths = sorted(list(pathlib.Path(self.currFolder).glob("*.jpg")))
      self.imagePaths += sorted(list(pathlib.Path(self.currFolder).glob("*.png")))
      print(self.imagePaths)
      if(self.imagePaths == []):
        QMessageBox.about(self, "Alert", "No .jpg or .png file found")
      else:
        self.maxIndex = len(self.imagePaths)-1
        # print(self.maxIndex,self.currIndex)
        self.changeImage(0)

    def nextImage(self):
      if(self.currIndex!=self.maxIndex):
        self.changeImage(self.currIndex+1)


    def prevImage(self):
      if(self.currIndex!=0):
        self.changeImage(self.currIndex-1)


    def changeImage(self,index):
      if(index>=0 and index<=self.maxIndex):
        pixmap = QPixmap(str(self.imagePaths[index]))
        width = min(pixmap.width(),1000)
        height = min(pixmap.width(),600)
        pixmap = pixmap.scaled(width,height,Qt.KeepAspectRatio)
        self.midLabel.setAlignment(Qt.AlignCenter)
        self.midLabel.setPixmap(pixmap)
        # self.resize(pixmap.width(),pixmap.height())
        self.currIndex = index

    def radioListner(self):
      radioButton = self.sender()
      if radioButton.isChecked(): 
        self.model_dir = radioButton.algo_dir

    def checkBoxListner(self):
      checkBox = self.sender()
      if(checkBox.isChecked()):
        self.req_classes +=[checkBox.label_class]
      else:
        self.req_classes.remove(checkBox.label_class)
      print(self.req_classes)

    def spinBoxListener(self):
      sp = self.sender()
      self.threshold = sp.value()
      self.threshold = float("{:.2f}".format(self.threshold))
      # print((self.threshold))

    def detectButtonclickListner(self):
      getInf = GetInference(self.model_dir,str(
        self.imagePaths[self.currIndex]),self.threshold,self.req_classes)
      img = getInf.show_inference()

      qim = ImageQt(img)
      pixmap = QPixmap.fromImage(qim)
      width = min(pixmap.width(),1000)
      height = min(pixmap.width(),600)
      pixmap = pixmap.scaled(width,height,Qt.KeepAspectRatio)
      self.midLabel.setAlignment(Qt.AlignCenter)
      self.midLabel.setPixmap(pixmap)




    

class GetInference:
  def __init__(self,model_path,image_path,threshold,req_classes):
    self.model_path = model_path 
    self.image_path = image_path
    self.threshold = threshold
    self.req_classes = req_classes
    self.load_model()


  def load_model(self):
    model_dir = pathlib.Path(self.model_path)/"saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    self.detection_model = model
    print("\n\nLOAD DONE")

  def run_inference_for_single_image(self, model, image):
      image = np.asarray(image)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis,...]

      # Run inference
      output_dict = model(input_tensor)
      # print("\n\nFirst this")
      # print(output_dict)
      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      num_detections = int(output_dict.pop('num_detections'))
      output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
      output_dict['num_detections'] = num_detections

      # detection_classes should be ints.
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
       
      # Handle models with masks:
      if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
      return output_dict

  def show_inference(self):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    model = self.detection_model
    image_path = self.image_path
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = self.run_inference_for_single_image(model, image_np)
    print(output_dict)
    # modifying output_dict



    filter = [1]*output_dict['num_detections']
    for i in range(output_dict['num_detections']):
      if(output_dict['detection_classes'][i] not in self.req_classes):
        filter[i] = 0
      if(output_dict['detection_scores'][i]<self.threshold):
        filter[i]=0
    filter_in=[]
    num_detections=output_dict.num_detections
    for i in range(output_dict['num_detections']):
      if(filter[i]==0):
        filter_in+=[i]
        num_detections-=1

    output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'],filter_in,0)
    output_dict['detection_classes'] = np.delete(output_dict['detection_classes'],filter_in)
    output_dict['detection_scores'] = np.delete(output_dict['detection_scores'],filter_in)
    output_dict['num_detections'] = num_detections
    print('\n\n',output_dict)



    # mod_boxes = np.ndarray(shape=(0,4),dtype = float)
    # print(type(output_dict['detection_classes']))
    # print(type(mod_boxes))
    # mod_classes = np.array(shape=(0,),dtype = int)
    # mod_scores = np.array(shape=(0,),dtype = float) 
    # for i in range(output_dict['num_detections']):
    #   if(filter[i]==1):
    #     mod_boxes = np.append(mod_boxes,output_dict['detection_boxes'][i])
    #     mod_classes = np.append(mod_classes,output_dict['detection_classes'][i])
    #     mod_scores = np.append(mod_scores,output_dict['detection_scores'][i])
        # mod_boxes+=[output_dict['detection_boxes'][i]]
        # mod_classes+=[output_dict['detection_classes'][i]]
        # mod_scores += [output_dict['detection_scores'][i]]
    # Visualization of the results of a detection.
    # print(mod_boxes,mod_classes,mod_scores)
    PATH_TO_LABELS = 'data/mscoco_label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     mod_boxes,
    #     mod_classes,
    #     mod_scores,
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks_reframed', None),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)
    outImage = Image.fromarray(image_np)
    # outImage.save(image_path+'out')
    return outImage



def get_main_app(argv=[]):


    app = QApplication(argv)
    app.setApplicationName(__appname__)


    win = MainWindow()
    # win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())
