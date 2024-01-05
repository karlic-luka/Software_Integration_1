import os, sys, math, pdb

## PyQt5 libraries
from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtCore import *
from PyQt5.QtOpenGL import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

## OpenGL libraries (pip install pyOpenGL)
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL import *
import OpenGL.GL as gl

## numpy libraries
import numpy as np
from numpy import linalg

## scipy libraries
from scipy import linalg

## imageio libraries (pip install imageio)
import imageio
from imageio.v2 import imsave
from imageio.v2 import imread

## Import the class "Ui_MainWindow" from the file "GUI.py"
from GUI_v2 import Ui_MainWindow
from pca_params_new_window import Ui_Form as PCA_Params_Ui_Form

## Import the class "OBJ" and "OBJFastV" from the file "OBJ.py"
from OBJ import OBJ

from pca_threads import TextureThreadClass, GeometryThreadClass

TEXTURE_WEIGHTS_MULTIPLICATIVE_FACTOR = 2
GEOMETRY_WEIGHTS_MULTIPLICATIVE_FACTOR = 1
# TODO add title, my name etc.
# TODO add RESET button to set all sliders to 0

class PCAParametersWindow(QWidget):
    def __init__(self, parent):
        super(PCAParametersWindow, self).__init__()

        self.parent = parent
        self.parent : MyMainWindow # type hinting
        QWidget.__init__(self)
        PCA_Params_Ui_Form.__init__(self)

        self.params_ui = PCA_Params_Ui_Form()
        self.params_ui.setupUi(self)

        # connect all sliders
        self.tex_sliders, self.geom_sliders = self.get_all_sliders()
        self.initialize_sliders()
        self.connect_tex_sliders()
        self.connect_geom_sliders()

        # connect buttons
        self.change_design()
        self.show()
        return
    
    def change_design(self):
        ### Colors/Design examples
        # Main Window
        # self.setStyleSheet('background-color: #333; color: white;')
        # # Common style for groupBox_2 and groupBox_3
        # group_box_style = (
        #     "QGroupBox {"
        #     "   border-radius: 9px;"
        #     "   border: 2px solid #888888;"
        #     "   margin-top: 0.5em;"
        #     "}"
        #     "QGroupBox:title {"
        #     "   padding: 0 6px 0 6px;"
        #     "   background-color: #333;"
        #     "   subcontrol-origin: margin;"
        #     "   subcontrol-position: top center;"
        #     "}"
        #     "font-weight: bold;"
        #     "font-size: 18px;"
        # )

        # # Apply the common style to both groupBox_2 and groupBox_3
        # self.params_ui.groupBox_2.setStyleSheet(group_box_style)
        # self.params_ui.groupBox_3.setStyleSheet(group_box_style)

        # Overall background color and text color for the main widget
        # self.setStyleSheet('background-color: #1E1E1E; color: #FFFFFF;')

        # # Common style for groupBox_2 and groupBox_3
        # group_box_style = (
        #     "QGroupBox {"
        #     "   border: 2px solid #3E3E3E;"
        #     "   border-radius: 10px;"
        #     "   margin-top: 0.5em;"
        #     "}"
        #     "QGroupBox:title {"
        #     "   subcontrol-origin: margin;"
        #     "   subcontrol-position: top center;"
        #     "   padding: 0 10px 0 10px;"
        #     "   background-color: #333333;"
        #     "   color: #FFFFFF;"
        #     "   font-weight: bold;"
        #     "   font-size: 18px;"
        #     "}"
        # )

        # # Apply the updated style to both groupBox_2 and groupBox_3
        # self.params_ui.groupBox_2.setStyleSheet(group_box_style)
        # self.params_ui.groupBox_3.setStyleSheet(group_box_style)

        # Overall background color and text color for the main widget
        # self.setStyleSheet('background-color: #F0F0F0; color: #333333;')

        # # Common style for groupBox_2 and groupBox_3
        # group_box_style = (
        #     "QGroupBox {"
        #     "   border: 2px solid #CCCCCC;"
        #     "   border-radius: 5px;"
        #     "   margin-top: 0.5em;"
        #     "}"
        #     "QGroupBox:title {"
        #     "   subcontrol-origin: margin;"
        #     "   subcontrol-position: top center;"
        #     "   padding: 5px 10px;"
        #     "   background-color: #E0E0E0;"
        #     "   color: #333333;"
        #     "   font-weight: bold;"
        #     "   font-size: 16px;"
        #     "}"
        # )

        # # Apply the updated style to both groupBox_2 and groupBox_3
        # self.params_ui.groupBox_2.setStyleSheet(group_box_style)
        # self.params_ui.groupBox_3.setStyleSheet(group_box_style)

        # Overall background color and text color for the main widget
        # self.setStyleSheet('background-color: #EDEDED; color: #333333;')

        # # Common style for groupBox_2 and groupBox_3
        # group_box_style = (
        #     "QGroupBox {"
        #     "   border: 2px solid #A0A0A0;"
        #     "   border-radius: 8px;"
        #     "   margin-top: 0.5em;"
        #     "   background-color: #FFFFFF;"
        #     "}"
        #     "QGroupBox:title {"
        #     "   subcontrol-origin: margin;"
        #     "   subcontrol-position: top center;"
        #     "   padding: 8px 15px;"
        #     "   background-color: #4CAF50;"  # Green title background
        #     "   color: #FFFFFF;"
        #     "   font-weight: bold;"
        #     "   font-size: 16px;"
        #     "}"
        # )

        # # Apply the updated style to both groupBox_2 and groupBox_3
        # self.params_ui.groupBox_2.setStyleSheet(group_box_style)
        # self.params_ui.groupBox_3.setStyleSheet(group_box_style)

        # Apply the updated style to both groupBox_2 and groupBox_3
        font = QFont("Roboto", 11)  # You can adjust the size as needed

        # Set the font for the main widget
        self.setFont(font)
        # Overall background color and text color for the main widget
        self.setStyleSheet('background-color: #EDEDED; color: #333333;')

        # Common style for groupBox_2 and groupBox_3
        group_box_style = (
            "QGroupBox {"
            "   border: 2px solid #4682B4;"  # Steel Blue border
            "   border-radius: 8px;"
            "   margin-top: 0.5em;"
            "   background-color: #FFFFFF;"
            "}"
            "QGroupBox:title {"
            "   subcontrol-origin: margin;"
            "   subcontrol-position: top center;"
            "   padding: 8px 15px;"
            "   background-color: #4682B4;"  # Steel Blue title background
            "   color: #FFFFFF;"
            "   font-weight: bold;"  # Make the title text bold
            "   font-size: 16px;"
            "}"
        )

        group_box_style = ("QGroupBox { border-radius: 9px; border: 2px solid #3498DB; margin-top: 0.5em; background-color: #85C1E9; }"
                         "QGroupBox:title { padding: 0 6px 0 6px; background-color: #3498DB; subcontrol-origin: margin; subcontrol-position: top center; color: white;}"
                         "#groupBox_3 { font-weight: bold; font-size: 18px; }")

        # Customize the style of QPushButton (buttons) with a lighter shade of blue
        button_style = ("QPushButton { background-color: #5DADE2; border: 2px solid #3498DB; color: white; border-radius: 5px; padding: 5px; }"
                       "QPushButton:hover { background-color: #5499C7; }")
        # Apply the updated style to both groupBox_2 and groupBox_3
        self.params_ui.groupBox_2.setStyleSheet(group_box_style)
        self.params_ui.groupBox_3.setStyleSheet(group_box_style)
        self.params_ui.pbReset_texture.setStyleSheet(button_style)
        self.params_ui.pbReset_geometry.setStyleSheet(button_style)
        return


    def initialize_sliders(self):
        print(f'Inside PCAParametersWindow.initialize_sliders')
        for slider in self.tex_sliders:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            slider.setEnabled(False)
        
        for slider in self.geom_sliders:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            slider.setEnabled(False)
        print(f'Initialized sliders')
        return
    
            
    def get_all_sliders(self):
        tex_sliders = []
        for i in range(1, 11):
            slider = getattr(self.params_ui, f'Tslider_{i}')
            tex_sliders.append(slider)
        geom_sliders = []
        for i in range(1, 23):
            slider = getattr(self.params_ui, f'Gslider_{i}')
            geom_sliders.append(slider)
        print(f'Got all sliders')
        print(f'Number of texture sliders: {len(tex_sliders)}')
        print(f'Number of geometry sliders: {len(geom_sliders)}')
        return tex_sliders, geom_sliders
    
    def connect_tex_sliders(self):
        for slider in self.tex_sliders:
            slider.valueChanged.connect(self.T_SliderValueChange)
            slider.setEnabled(False)
        return
    
    def connect_geom_sliders(self):
        for slider in self.geom_sliders:
            slider.valueChanged.connect(self.G_SliderValueChange)
            slider.setEnabled(False)
        return
    
    def T_SliderValueChange(self, value):
        # TODO prebaciti mozda ovdje jer mi se cini kao da se zablokira GUI kada se ovo pozove
        # print(f'Inside PCAParametersWindow.T_SliderValueChange')
        # change label
        self.change_texture_label(value, self.sender())
        self.parent.T_SliderValueChange(value)
        return
    
    def G_SliderValueChange(self, value):
        # TODO prebaciti mozda ovdje jer mi se cini kao da se zablokira GUI kada se ovo pozove
        # print(f'Inside PCAParametersWindow.G_SliderValueChange')
        self.change_geometry_label(value, self.sender())
        self.parent.G_SliderValueChange(value)
        return
    
    def change_geometry_label(self, value, slider):
        try:
            slider_id = int(slider.objectName().split('_')[-1])
            geometry_comp_label = getattr(self.params_ui, f'geom_comp_label_{slider_id}')
            percentage = np.abs((value - slider.minimum()) / (slider.maximum() - slider.minimum()) * 100)
            geometry_comp_label_text = f'{value} = {percentage : .1f}%'
            geometry_comp_label.setText(geometry_comp_label_text)
        except Exception as e:
            print(f'Error updating geometry slider label: {e}')
            print(f'Geometry weight value was probably changed by the program, not by the user.')
        return
    
    def change_texture_label(self, value, slider):
        try:
            slider_id = int(slider.objectName().split('_')[-1])
            texture_comp_label = getattr(self.params_ui, f'tex_comp_label_{slider_id}')
            percentage = np.abs((value - slider.minimum()) / (slider.maximum() - slider.minimum()) * 100)
            texture_comp_label_text = f'{value} = {percentage : .1f}%'
            texture_comp_label.setText(texture_comp_label_text)
        except Exception as e:
            print(f'Error updating texture slider label: {e}')
            print(f'Texture weight value was probably changed by the program, not by the user.')
        return


    

####################################################################################################
# The Main Window (GUI) --- TASKS TO DO HERE
####################################################################################################
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)  # The 2 lines here are always presented like this
        QMainWindow.__init__(self, parent)  # Just to initialize the window

        # All the elements from our GUI are added in "ui"
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.params_window = PCAParametersWindow(parent=self)

        # Default param
        self.InputModelLoaded = False
        self.InputTextureLoaded = False
        self.InputListCreated = False
        self.InputTexturePath = []
        self.InputModel = []
        self.InputTextureDim = 256

        self.TargetModelLoaded = False
        self.TargetTextureLoaded = False
        self.TargetListCreated = False
        self.TarTexturePath = []
        self.TarModel = []

        self.bg_color = 0.0
        self.Root = {}
        self.Tval = self.Gval = 0
        self.P_Tval = self.P_Gval = 0
        self.Tx = self.Ty = 0
        self.Tz = 1
        self.r_mode = self.c_mode = "Faces"
        self.bg_color = 0.0
        self.LeftXRot = self.LeftYRot = 0
        self.b_Ready = False
        self.Updated = False
        self.b_ProcessDone = self.b_Process2Done = self.b_Ready = self.PCA_done = False
        self.old_Gval = self.old_Tval = 0

        # Add a GLWidget (will be used to display our 3D object)
        self.glWidget = GLWidget(parent=self)
        # Add the widget in "frame_horizontalLayout", an element from the GUI
        self.ui.frame_horizontalLayout.addWidget(self.glWidget)

        # Update Widgets
        # Connect a signal "updated" between the GLWidget and the GUI, just to have a link between the 2 classes
        self.glWidget.updated.connect(self.updateFrame)

        # RadioButton (Rendering Mode)
        # Connect the radiobutton to the function on_rendering_button_toggled
        self.ui.rbFaces.toggled.connect(self.rendering_button_toggled)
        # It will be used to switch between 2 modes, full/solid model or cloud of points
        self.ui.rbPoints.toggled.connect(self.rendering_button_toggled)

        # RadioButton (Background Color)
        # Connect the radiobutton to the function on_bgcolor_button_toggled
        self.ui.rbWhite.toggled.connect(self.bgcolor_button_toggled)
        # Just an example to change the background of the 3D frame
        self.ui.rbBlack.toggled.connect(self.bgcolor_button_toggled)

        # Buttons
        # Connect the button to the function LoadFileClicked (will read the 3D file)
        self.ui.LoadFile.clicked.connect(self.LoadFileClicked)
        # Connect the button to the function ProcessClicked (will process PCA)
        self.ui.Process.clicked.connect(self.ProcessClicked)
        # Connect the button to the function SaveOBJ (will write a 3D file)
        self.ui.exportResult.clicked.connect(self.SaveOBJ)
        self.ui.pb_stop_processing.clicked.connect(self.stop_threads)

        # connect threads
        self.finished_threads_counter = 0
        self.pca_texture_thread = TextureThreadClass()
        self.pca_texture_thread.finished.connect(self.PCA_Tex)
        self.pca_texture_thread.updated.connect(self.update_texture_progress_bar)
        self.pca_geometry_thread = GeometryThreadClass()
        self.pca_geometry_thread.finished.connect(self.PCA_Geo)
        self.pca_geometry_thread.updated.connect(self.update_geometry_progress_bar)

        ### Colors/Design examples
        # Main Window
        self.setup_styles()
        self.setStyleSheet('background-color: #333; color: white;')

        return
    
    def setup_styles(self):
        # Set the background color of the main window
        # self.ui.centralwidget.setStyleSheet('background-color: #333; color: white;')
        self.ui.centralwidget.setStyleSheet('background-color: #EDEDED; color: #333333;')

        # group_box_style = ("QGroupBox { border-radius: 9px; border: 2px solid #3498DB; margin-top: 0.5em; background-color: #85C1E9; }"
        #                  "QGroupBox:title { padding: 0 6px 0 6px; background-color: #3498DB; subcontrol-origin: margin; subcontrol-position: top center; color: white;}"
        #                  "#groupBox_3 { font-weight: bold; font-size: 18px; }")
        
        # # Customize the style of QPushButton (buttons) with a lighter shade of blue
        # buttonStyle = ("QPushButton { background-color: #5DADE2; border: 2px solid #3498DB; color: white; border-radius: 5px; padding: 5px; }"
        #                "QPushButton:hover { background-color: #5499C7; }")
        group_box_style = ("QGroupBox { border-radius: 9px; border: 2px solid #3498DB; margin-top: 0.5em; background-color: #85C1E9; }"
                         "QGroupBox:title { padding: 0 6px 0 6px; background-color: #3498DB; subcontrol-origin: margin; subcontrol-position: top center; color: white;}"
                         "#groupBox_3 { font-weight: bold; font-size: 18px; }")

        # Customize the style of QPushButton (buttons) with a lighter shade of blue
        buttonStyle = ("QPushButton { background-color: #5DADE2; border: 2px solid #3498DB; color: white; border-radius: 5px; padding: 5px; }"
                       "QPushButton:hover { background-color: #5499C7; }")
        # Apply the style to the buttons
        self.ui.LoadFile.setStyleSheet(buttonStyle)
        self.ui.Process.setStyleSheet(buttonStyle)
        self.ui.exportResult.setStyleSheet(buttonStyle)
        self.ui.pb_stop_processing.setStyleSheet(buttonStyle)
        

        # Apply the style to a specific QGroupBox (replace groupBox_3 with the actual name of your group box)
        self.ui.groupGUI.setStyleSheet(group_box_style)
        return
    
    def closeEvent(self, event):
        print(f'Inside closeEvent')
        if hasattr(self, 'pca_texture_thread'):
            if self.pca_texture_thread.isRunning():
                self.pca_texture_thread.requestInterruption()
                self.pca_texture_thread.stop()
                self.pca_texture_thread.quit()

        if hasattr(self, 'pca_geometry_thread'):
            if self.pca_geometry_thread.isRunning():
                self.pca_geometry_thread.requestInterruption()
                self.pca_geometry_thread.stop()
                self.pca_geometry_thread.quit()
        return
    
    def stop_threads(self):
        print(f'Manually stopping threads')
        if hasattr(self, 'pca_texture_thread'):
            if self.pca_texture_thread.isRunning():
                self.pca_texture_thread.requestInterruption()

        if hasattr(self, 'pca_geometry_thread'):
            if self.pca_geometry_thread.isRunning():
                self.pca_geometry_thread.requestInterruption()

        return
    
    def update_texture_progress_bar(self, value):
        self.ui.progress_texture.setValue(value)
        return
    
    def update_geometry_progress_bar(self, value):
        self.ui.progress_geometry.setValue(value)
        return

    
    def LoadFileClicked(self):
        try:
            # To display a popup window that will be used to select a file (.obj or .png)
            # The .obj and .png should have the same name!
            self.myFile = QFileDialog.getOpenFileName(None, 'OpenFile', "", "3D object(*.obj);;Texture(*.png)")
            self.myPath = self.myFile[0]
            # If the extension is .obj (or .png), will remove the 4 last characters (== the extension)
            self.GlobalNameWithoutExtension = self.myPath[:-4]
            self.FileNameWithExtension = QFileInfo(self.myFile[0]).fileName()  # Just the filename
            if self.myFile[0] == self.myFile[1] == '':
                # No file selected or cancel button clicked - so do nothing
                pass
            else:
                self.InputModel = self.TarModel = []
                self.InputModelLoaded = self.InputTextureLoaded = self.InputListCreated = False
                self.InputTexturePath = self.GlobalNameWithoutExtension + ".png"

                # Will use the class OBJ to read our 3D file and extract everything
                self.InputModel = OBJ(self.GlobalNameWithoutExtension + ".obj")

                imsave("TarTexture" + ".png", imread(self.InputTexturePath))

                self.TarTexturePath = '/'.join(self.myPath.split('/')[:-1]) + '/TarTexture.png'
                self.TarModel = self.InputModel

                # We read the 2 files, so we can now set the boolean value to True
                # (the GLWidget will now display it automatically because of the 2 variables used there)
                self.InputModelLoaded = self.InputTextureLoaded = True
                self.PCA_done = False
                self.glWidget.update()

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print(self.myFile)
        except ValueError:
            print("Value Error.")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    
    def ProcessClicked(self):

        self.pca_texture_thread.start()
        self.pca_geometry_thread.start()
        return

    def on_PCA_is_finished(self):
        self.PCA_done = True
        # Unlock and prepare sliders
        # self.ui.Tslider.blockSignals(True)
        # self.ui.Tslider.setValue(0)
        # self.T_SliderValueChange(0)
        # self.ui.Tslider.blockSignals(False)
        # self.ui.Tslider.setEnabled(True)
        # # S = self.checkSign(round(self.Root['Tex']['WTex'][0]), round(self.Root['Tex']['WTex'][1]))
        # # Tmin = round (S * self.Root['Tex']['WTex'][0])
        # # Tmax = round (S * self.Root['Tex']['WTex'][1])
        # Tmin, Tmax = -255, 255
        # self.ui.Tslider.setRange(Tmin, Tmax)

        # self.ui.Gslider.blockSignals(True)
        # self.ui.Gslider.setValue(0)
        # self.G_SliderValueChange(0)
        # self.ui.Gslider.blockSignals(False)
        # self.ui.Gslider.setEnabled(True)
        # # Gmin = round(self.Root['models']['WGeo'][0])
        # # Gmax = round(self.Root['models']['WGeo'][1])
        # Gmin, Gmax = -255, 255
        # self.ui.Gslider.setRange(Gmin, Gmax)

        print(f'Inside on_PCA_is_finished')
        print(f'Updating Texture sliders')
        # update texture sliders
        tex_initialized = False
        for slider_id, slider in enumerate(self.params_window.tex_sliders):
            # print(f'Updating slider with id={slider_id}')
            slider.blockSignals(True)
            slider.setValue(0)
            if not tex_initialized:
                self.params_window.T_SliderValueChange(0)
                tex_initialized = True
            slider.blockSignals(False)
            slider.setEnabled(True)
            min_temp = self.Root['Tex']['WTex'][0][slider_id]
            max_temp = self.Root['Tex']['WTex'][1][slider_id]
            S = self.checkSign(round(min_temp), round(max_temp))
            Tmin = round(S * min_temp)
            Tmax = round(S * max_temp)
            slider.setRange(Tmin, Tmax)

        print(f'Updating Geometry sliders')
        geometry_initialized = False
        # update geometry sliders
        for slider_id, slider in enumerate(self.params_window.geom_sliders):
            # print(f'Updating slider with id={slider_id}')
            slider.blockSignals(True)
            slider.setValue(0)
            if not geometry_initialized:
                self.params_window.G_SliderValueChange(0)
                geometry_initialized = True
            slider.blockSignals(False)
            slider.setEnabled(True)
            Gmin = round(self.Root['models']['WGeo'][0][slider_id])
            Gmax = round(self.Root['models']['WGeo'][1][slider_id])
            slider.setRange(Gmin, Gmax)
        return
    
    def PCA_Tex(self, result: dict):

        ###########################################
        # TASK 1 FOR THE ASSIGNMENT
        ###########################################

        ## Note: You will be evaluated on each task
        ## If it doesn't work, you will be evaluated on your efforts and the quality of the code by trying to do it

        ## Guideline: Read model1.png and model2.png
        ## Do the PCA with the 2 textures, similar to what we did in session 5
        eigenvectors_transposed_flattened = result['eigenvectors_transposed_flattened']
        texture_mean = result['mean']
        texture_weights = result['weights']
        try:
            ## Instead of saving in a yml file, we will save a structure
            ## The variable self.Root['Tex'] is related to the texture
            self.Root['Tex'] = {}

            ## Save results
            self.Root['Tex']['VrTex'] = eigenvectors_transposed_flattened  # eigenvector variable (transpose/flatten)
            print(f'Texture transposed/flattened eigenvectors shape: {self.Root["Tex"]["VrTex"].shape}')
            self.Root['Tex']['XmTex'] = texture_mean # average texture variable
            print(f'Texture mean shape: {self.Root["Tex"]["XmTex"].shape}')
            texture_weights *= TEXTURE_WEIGHTS_MULTIPLICATIVE_FACTOR
            print(f'Some texture weights: {texture_weights[0, :5]}')
            # save min and max weights for each row
            temp_minimums = np.min(texture_weights, axis=1)
            temp_maximums = np.max(texture_weights, axis=1)
            self.Root['Tex']['WTex'] = [temp_minimums, temp_maximums]
            print(f'Texture min shape: {self.Root["Tex"]["WTex"][0].shape}')
            print(f'Texture max shape: {self.Root["Tex"]["WTex"][1].shape}')
            print(f'Texture min and max weights: {self.Root["Tex"]["WTex"]}')
            # self.Root['Tex']['WTex'] =  [texture_weights.min(), texture_weights.max()] # min and max weights : format : [min, max]

        except Exception as e:
            print('PCA_Tex Error:', e)
            return
        self.b_ProcessDone = True
        self.finished_threads_counter += 1
        if self.finished_threads_counter == 2:
            self.on_PCA_is_finished()
        print(f'COUNTER: {self.finished_threads_counter}')
        return


    def PCA_Geo(self, result: dict):

        ###########################################
        # TASK 2 FOR THE ASSIGNMENT
        ###########################################

        ## Guideline: Read model1.obj and model2.obj with the "OBJFastV(...)" function to extract quickly the vertices
        ## Do the PCA with the vertices (similar to the function PCA_Tex(), try to do the same but with the vertices)
        ## Or adapt the code a bit
        eigenvectors_transposed = result['eigenvectors_transposed']
        geometry_mean = result['mean']
        geometry_weights = result['weights']
        try:
            # Save results
            ## Instead of saving in a yml file, we will save a structure
            ## The variable self.Root['models'] is related to the geometry
            self.Root['models'] = {}

            ## Save results
            self.Root['models']['VrGeo'] = eigenvectors_transposed  # eigenvector variable (transpose)
            print(f'Geometry transposed eigenvectors shape: {self.Root["models"]["VrGeo"].shape}')
            self.Root['models']['XmGeo'] = geometry_mean  # average texture variable
            print(f'Geometry mean shape: {self.Root["models"]["XmGeo"].shape}')
            geometry_weights *= GEOMETRY_WEIGHTS_MULTIPLICATIVE_FACTOR
            print(f'Some geometry weights: {geometry_weights[0, :5]}')
            temp_minimums = np.min(geometry_weights, axis=1)
            temp_maximums = np.max(geometry_weights, axis=1)
            self.Root['models']['WGeo'] = [temp_minimums, temp_maximums]
            # self.Root['models']['WGeo'] =  [geometry_weights.min(), geometry_weights.max()]  # min and max weights : format : [min, max]

        except Exception as e:
            print('PCA_Geo Error:', e)
            return
        self.b_Process2Done = True
        print(f'PCA_Geo DONE.')
        self.finished_threads_counter += 1
        print(f'COUNTER: {self.finished_threads_counter}')
        if self.finished_threads_counter == 2:
            self.on_PCA_is_finished()
        return


    def T_SliderValueChange(self, value):
        self.Tval = value
        # print(f'Tval: {self.Tval}')
        # collect all weights from sliders
        self.texture_slider_weights = []
        for i in range(len(self.params_window.tex_sliders)):
            self.texture_slider_weights.append(self.params_window.tex_sliders[i].value())
        self.texture_slider_weights = np.array(self.texture_slider_weights)
        # print(f'Tval: {self.texture_slider_weights}')


        ###########################################
        # TASK 3 FOR THE ASSIGNMENT
        ###########################################

        ## Texture slider ( PCA : Input Texture <----> Target Texture )
        if self.b_ProcessDone == True and self.b_Process2Done == True:
            ## You have to create the new texture by uGsing the formula of the PCA
            ## As a reminder, Texture = Mean + W1 * E1 + W2 * E2 + ....
            ## Mean is: self.Root['Tex']['XmTex']
            ## W is: Tval  (weight is linked to the slider value "Tval")
            ## E is: self.Root['Tex']['VrTex'][0]
            ## The product Wi * Ei must be done using np.dot(value1, value2)

            try:

                ## >>> ADD CODE BELOW <<<
                # print(f'Part of the texture mean: {self.Root["Tex"]["XmTex"][:5]}')
                self.N_TarTex = self.Root['Tex']['XmTex'].copy() # NOTE: without copy() it doesn't work!!! POST MORTEM: it's because of the reference
                for i in range(len(self.params_window.tex_sliders)):
                    self.N_TarTex += np.dot(self.texture_slider_weights[i], self.Root['Tex']['VrTex'][i].copy()) # NOTE: without copy() it doesn't work!!! POST MORTEM: it's because of the reference

            except Exception as e:
                print('New target texture Error', e)

            ## Reshape the variable self.N_TarTex to have the real texture size 256*256*4 (RGBA) instead of 1D array
            ## Convert astype np.uint8 to be usable
            try:

                ## >>> ADD CODE BELOW <<<

                self.TarTexture = np.reshape(self.N_TarTex, (256, 256, 4))
                self.TarTexture[self.TarTexture < 0] = 0
                self.TarTexture[self.TarTexture > 1] = 1
                self.TarTexture = (self.TarTexture * 255).astype(np.uint8)
            except Exception as e:
                print('TarTexture Error:', e)

            # Save the new texture
            try:
                imageio.v2.imsave("TarTexture" + ".png", self.TarTexture)
            except Exception as e:
                print('TarTexture Save error:', e)

        return

    def G_SliderValueChange(self, value):
        # collect all weights from sliders
        self.Gval = value
        self.geom_slider_weights = []
        for i in range(len(self.params_window.geom_sliders)):
            self.geom_slider_weights.append(self.params_window.geom_sliders[i].value())
        self.geom_slider_weights = np.array(self.geom_slider_weights)
        # print(f'geom_slider_weights: {self.geom_slider_weights}')


        ###########################################
        # TASK 4 FOR THE ASSIGNMENT
        ###########################################

        ## Geometry slider ( model1 <----> model2 )
        if self.b_ProcessDone == True and self.b_Process2Done == True:
            ## You have to create the new geometry by using the formula of the PCA
            ## As a reminder, 3DFace = Mean + W1 * E1 + W2 * E2 + ....
            ## Mean is: self.Root['models']['XmGeo']
            ## W is: Gval  (weight is linked to the slider value "Gval")
            ## E is: self.Root['models']['VrGeo']
            ## The product W * E must be done using np.dot(value1, value2)
            try:

                ## >>> ADD CODE BELOW <<<
                # print(f'Part of the geometry mean: {self.Root["models"]["XmGeo"][:5]}')
                self.N_TarModel = self.Root['models']['XmGeo'].copy() # NOTE: without copy() it doesn't work!!! POST MORTEM: it's because of the reference
                for i in range(len(self.params_window.geom_sliders)):
                    self.N_TarModel += np.dot(self.geom_slider_weights[i], self.Root['models']['VrGeo'][i].copy()) # NOTE: without copy() it doesn't work!!! POST MORTEM: it's because of the reference

            except Exception as e:
                print('New target model', e)

            ## Reshape the variable self.N_TarModel to have the real geometry format : x y z
            ## For info, the 1D array is like this: x x x x ... y y y y ... z z z z (5904 values 3 times)
            ## The first 5904 values in self.N_TarModel are X, next 5904 are Y, last 5904 are Z
            ## Final format should be : 5904 rows and 3 columns :
            ## x y z
            ## x y z
            ## ....

            arr_3d = np.zeros((5904, 3))
            ## Use for loops to reconstruct completely and save the new vertices in self.TarModel.vertices
            ## Do not forget to concatenate x, y and z together (as a float)

            ## >>> ADD CODE BELOW <<<
            for i in range(5904):
                arr_3d[i, 0] = self.N_TarModel[i] 
                arr_3d[i, 1] = self.N_TarModel[i + 5904]
                arr_3d[i, 2] = self.N_TarModel[i + 2*5904]
                
            row = temp = []
            for i in range(5904):
                row = float(arr_3d[i, 0]), float(arr_3d[i, 1]), float(arr_3d[i, 2])
                temp.append(row)
            ## self.TarModel.vertices is the new 3D model
            try:
                self.TarModel.vertices = temp
            except Exception as e:
                print(e)

    def SaveOBJ(self):
        ###########################################
        # TASK 5 FOR THE ASSIGNMENT
        ###########################################

        ## Try to see how the OBJ class works (file OBJ.py)
        ## Instead of reading, you should now write in a file
        ## You will need to add everything in the .obj, not only vertices!

        ## Open the model1.obj with a text editor for example to see what you have in it
        ## The new file will be similar but with the new vertices (v)
        ## You can reuse the original file to add back vt, vn, f...

        ## >>> ADD CODE BELOW <<<
        try:
            with open(self.GlobalNameWithoutExtension + ".obj", 'r') as file:
                original_lines = file.readlines()
            file.close()

            temp_vertices = self.TarModel.vertices.copy()
            if len(temp_vertices) != 5904:
                print('Error: Wrong number of vertices')
                return
            
            for idx, line in enumerate(original_lines):
                if line.startswith('v '): # space after v is important to avoid matching vt
                    row = temp_vertices.pop(0)
                    original_lines[idx] = f'v {row[0]} {row[1]} {row[2]}\n'

            newfile_name = self.GlobalNameWithoutExtension + "_new_model.obj"
            with open(newfile_name, 'w') as file:
                file.writelines(original_lines)
            print(f'Saved new model to {newfile_name}')
            file.close()
                
        except Exception as e:
            print(f'Error writing new model: {e}')
        return

    def checkSign(self, W1, W2):
        ## Check the weights, to know which one is negative/positive
        ## Important for the sliders to have the - on the left and + on the right
        if W1 < 0:
            res = 1
        else:
            res = -1
        return res



    def rendering_button_toggled(self):
        radiobutton = self.sender()

        if radiobutton.isChecked():
            self.r_mode = radiobutton.text()  # Save "Faces" or "Points" in r_mode
        self.Updated = True
        self.glWidget.update()

    def bgcolor_button_toggled(self):
        radiobutton = self.sender()  # Catch the click
        if radiobutton.isChecked():  # Will check which button is checked
            # Will store and use the text of the radiobutton
            # to store a value in the variable "bg_color" that will be used in the GLWidget
            color = radiobutton.text()
            if color == "White":
                self.bg_color = 1.0
            elif color == "Black":
                self.bg_color = 0.0

    def updateFrame(self):
        self.glWidget.update()

####################################################################################################
# The OpenGL Widget --- it's normally not needed to touch this part especially paintGL
####################################################################################################
class GLWidget(QGLWidget):
    updated = pyqtSignal(int)  # pyqtSignal is used to allow the GUI and the OpenGL widget to sync
    xRotationChanged = pyqtSignal(int)
    yRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)

    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.lastPos = QPoint()

        self.Tx = self.Ty = 0
        self.Tz = 1
        self.LeftXRot = self.LeftYRot = 0

        self.parent = parent

        self.InputListCreated = False
        self.TargetListCreated = False

    def initializeGL(self):
        glEnable(GL_TEXTURE_2D)
        self.tex = glGenTextures(1)

    def paintGL(self):

        self.InputModelLoaded = self.parent.InputModelLoaded
        self.InputTextureLoaded = self.parent.InputTextureLoaded
        self.InputTexturePath = self.parent.InputTexturePath
        self.InputModel = self.parent.InputModel
        self.InputTextureDim = self.parent.InputTextureDim

        self.TargetModelLoaded = self.parent.TargetModelLoaded
        self.TargetTextureLoaded = self.parent.TargetTextureLoaded
        self.TarTexturePath = self.parent.TarTexturePath
        self.TarModel = self.parent.TarModel

        self.bg_color = self.parent.bg_color
        self.Root = self.parent.Root
        self.Tval = self.parent.Tval
        self.Gval = self.parent.Gval
        self.P_Tval = self.parent.P_Tval
        self.P_Gval = self.parent.P_Gval

        self.r_mode = self.parent.r_mode
        self.c_mode = self.parent.c_mode
        self.bg_color = self.parent.bg_color
        self.b_Ready = self.parent.b_Ready
        self.Updated = self.parent.Updated
        self.b_ProcessDone = self.parent.b_ProcessDone
        self.b_Process2Done = self.parent.b_Process2Done
        self.PCA_done = self.parent.PCA_done
        self.old_Gval = self.parent.old_Gval
        self.old_Tval = self.parent.old_Tval

        # If we have nothing to display, no model loaded: just a default background with axis
        if not self.InputModelLoaded:
            glClearColor(self.bg_color, self.bg_color, self.bg_color, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()  # identity matrix, resets the matrix back to its default state

            # field of view (angle), ratio, near plane, far plane: all values must be > 0
            gluPerspective(60, self.aspect, 0.01, 10000)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            glTranslate(self.Tx, self.Ty, -self.Tz)

            glRotated(self.xRot / 16, 1.0, 0.0, 0.0)
            glRotated(self.yRot / 16, 0.0, 1.0, 0.0)
            glRotated(self.zRot / 16, 0.0, 0.0, 1.0)

            self.qglColor(Qt.red)
            self.renderText(10, 20, "X")
            self.qglColor(Qt.green)
            self.renderText(10, 40, "Y")
            self.qglColor(Qt.blue)
            self.renderText(10, 60, "Z")

            glLineWidth(2.0)  # Width of the lines
            # To start creating lines (you also have glBegin(GL_TRIANGLES), glBegin(GL_POLYGONES), etc....
            # depending on what you want to draw)
            glBegin(GL_LINES)
            # X axis (red)
            glColor3ub(255, 0, 0)
            glVertex3d(0, 0, 0)  # The first glVertex3d is the starting point and the second the end point
            glVertex3d(1, 0, 0)
            # Y axis (green)
            glColor3ub(0, 255, 0)
            glVertex3d(0, 0, 0)
            glVertex3d(0, 1, 0)
            # Z axis (blue)
            glColor3ub(0, 0, 255)
            glVertex3d(0, 0, 0)
            glVertex3d(0, 0, 1)
            glEnd()  # Stop
            glLineWidth(1.0)  # Change back the width to default if you want to draw something else after normally

        else:
            PCA_done = self.parent.PCA_done
            # If a model is loaded but PCA is not done, display only the model
            if PCA_done == False:
                # display input 3D model
                if self.InputModelLoaded == True and self.InputTextureLoaded == True:
                    self.updated.emit(1)
                    glClearColor(self.bg_color, self.bg_color, self.bg_color, 1.0)
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()  # identity matrix, resets the matrix back to its default state
                    # field of view (angle), ratio, near plane, far plane, all values must be > 0
                    gluPerspective(60, self.aspect, 0.01, 10000)
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()
                    glTranslate(self.Tx, self.Ty, -self.Tz)
                    glRotated(self.xRot / 16, 1.0, 0.0, 0.0)
                    glRotated(self.yRot / 16, 0.0, 1.0, 0.0)
                    glRotated(self.zRot / 16, 0.0, 0.0, 1.0)
                    # Move 3D object to center
                    glPushMatrix()  # Save any translate/scale/rotate operations that you previously used
                    # In InputModel.vertices you have the coordinates of the vertices (X,Y,Z)
                    InputModel_Xs = [row[0] for row in self.InputModel.vertices]  # Here you will extract X
                    InputModel_Ys = [row[1] for row in self.InputModel.vertices]  # Here you will extract Y
                    InputModel_Zs = [row[2] for row in self.InputModel.vertices]  # Here you will extract Z
                    # A 3D object can have coordinates not always centered on 0
                    # So we are calculating u0,v0,w0 (center of mass/gravity of the 3D model)
                    # To be able to move it after to the center of the scene
                    u0 = (min(InputModel_Xs) + max(InputModel_Xs)) / 2
                    v0 = (min(InputModel_Ys) + max(InputModel_Ys)) / 2
                    w0 = (min(InputModel_Zs) + max(InputModel_Zs)) / 2
                    # Here we are calculating the best zoom factor by default (to see the 3D model entirely)
                    d1 = max(InputModel_Xs) - min(InputModel_Xs)
                    d2 = max(InputModel_Ys) - min(InputModel_Ys)
                    d3 = max(InputModel_Zs) - min(InputModel_Zs)
                    Q = 0.5 / ((d1 + d2 + d3) / 3)
                    glScale(Q, Q, Q)
                    glTranslate(-u0, -v0, -w0)  # Move the 3D object to the center of the scene
                    # Display 3D Model via a CallList (GOOD, extremely fast!)
                    # If the list is not created, we will do it
                    if self.InputModelLoaded == True and self.InputTextureLoaded == True and self.InputListCreated == False:
                        # pdb.set_trace()
                        ## This is how to set up a display list, whose invocation by glCallList
                        self.glinputModel = glGenLists(1)  # Allocate one list into memory
                        glNewList(self.glinputModel, GL_COMPILE)  # Begin building the passed in list
                        self.addTexture(self.InputTexturePath)  # Call function to add texture
                        self.addModel(self.InputModel)  # Call function to add 3D model
                        glEndList()  # Stop list creation
                        self.InputListCreated = True
                        self.c_mode = self.r_mode
                        glCallList(self.glinputModel)  # Call the list (display the model)
                    # If the list is already created, no need to process again and loose time, just display it
                    elif self.InputModelLoaded == True and self.InputTextureLoaded == True and self.InputListCreated == True:
                        # however, if we are changing the mode (Faces/Points), we need to recreate again the list
                        if self.Updated == True:
                            # Here we have to create the list again because it's not exactly the same list
                            # if we want to show just the points or the full model
                            self.glinputModel = glGenLists(1)
                            glNewList(self.glinputModel, GL_COMPILE)
                            self.addTexture(self.InputTexturePath)
                            self.addModel(self.InputModel)
                            glEndList()
                            self.c_mode = self.r_mode
                            glCallList(self.glinputModel)
                            self.Updated = False
                            self.parent.Updated = False
                        else:
                            glCallList(self.glinputModel)
                    glPopMatrix()  # Will reload the old model view matrix
                else:
                    print(0)

            # If the PCA is done, we will display the new model here
            else:

                glClearColor(self.bg_color, self.bg_color, self.bg_color, 1.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                glMatrixMode(GL_PROJECTION)
                glLoadIdentity()  # identity matrix, resets the matrix back to its default state
                gluPerspective(60, self.aspect, 0.01, 10000)
                glMatrixMode(GL_MODELVIEW)
                glLoadIdentity()
                glTranslate(self.Tx, self.Ty, -self.Tz)
                glRotated(self.xRot / 16, 1.0, 0.0, 0.0)
                glRotated(self.yRot / 16, 0.0, 1.0, 0.0)
                glRotated(self.zRot / 16, 0.0, 0.0, 1.0)
                # Move 3D object to center
                glPushMatrix()  # Save any translate/scale/rotate operations that you previously used
                # In InputModel.vertices you have the coordinates of the vertices (X,Y,Z), here you will extract X
                InputModel_Xs = [row[0] for row in self.InputModel.vertices]
                InputModel_Ys = [row[1] for row in self.InputModel.vertices]  # Here you will extract Y
                InputModel_Zs = [row[2] for row in self.InputModel.vertices]  # Here you will extract Z
                u0 = (min(InputModel_Xs) + max(InputModel_Xs)) / 2
                v0 = (min(InputModel_Ys) + max(InputModel_Ys)) / 2
                w0 = (min(InputModel_Zs) + max(InputModel_Zs)) / 2
                # Here we are calculating the best zoom factor by default (to see the 3D model entirely)
                d1 = max(InputModel_Xs) - min(InputModel_Xs)
                d2 = max(InputModel_Ys) - min(InputModel_Ys)
                d3 = max(InputModel_Zs) - min(InputModel_Zs)
                Q = 0.5 / ((d1 + d2 + d3) / 3)
                glScale(Q, Q, Q)
                glTranslate(-u0, -v0, -w0)  # Move the 3D object to the center of the scene
                self.setXRotation(self.LeftXRot)
                self.setYRotation(self.LeftYRot)
                self.updated.emit(1)
                if self.TargetListCreated == False:
                    self.targetModel = glGenLists(1)
                    glNewList(self.targetModel, GL_COMPILE)
                    self.applyTarTexture(self.parent.TarTexture)
                    self.addModel(self.InputModel)
                    glEndList()
                    self.TargetListCreated = True
                    self.c_mode = self.r_mode
                    glCallList(self.targetModel)
                elif self.TargetListCreated == True:
                    if self.c_mode == self.r_mode and self.old_Gval == self.Gval and self.old_Tval == self.Tval:
                        glCallList(self.targetModel)
                    else:
                        self.targetModel = glGenLists(1)
                        glNewList(self.targetModel, GL_COMPILE)
                        self.applyTarTexture(self.parent.TarTexture)
                        self.addModel(self.InputModel)
                        glEndList()
                        self.c_mode = self.r_mode
                        glCallList(self.targetModel)
                self.old_Gval = self.Gval
                self.old_Tval = self.Tval
                glPopMatrix()

    def addModel(self, InputModel):
        if self.r_mode == "Faces":
            glEnable(GL_TEXTURE_2D)
            glEnable(GL_DEPTH_TEST)  # to show all faces
            # glEnable(GL_CULL_FACE) # To hide non visible faces
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glBegin(GL_TRIANGLES)
            for i in InputModel.faces:
                F = i[0]
                for j in F:
                    glColor3ub(255, 255, 255)
                    glTexCoord2f(InputModel.texcoords[j-1][0], InputModel.texcoords[j-1][1])
                    glNormal3d(InputModel.normals[j-1][0], InputModel.normals[j-1][1], InputModel.normals[j-1][2])
                    glVertex3d(InputModel.vertices[j-1][0], InputModel.vertices[j-1][1], InputModel.vertices[j-1][2])
            glEnd()
            glDisable(GL_TEXTURE_2D)
        elif self.r_mode == "Points":
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glBegin(GL_POINTS)
            for i in range(len(InputModel.vertices)):
                glColor3ub(255, 255, 255)
                glTexCoord2f(InputModel.texcoords[i][0], InputModel.texcoords[i][1])
                glNormal3d(InputModel.normals[i][0], InputModel.normals[i][1], InputModel.normals[i][2])
                glVertex3d(int(InputModel.vertices[i][0]), int(InputModel.vertices[i][1]), int(InputModel.vertices[i][2]))
            glEnd()
            glDisable(GL_TEXTURE_2D)

    def addTexture(self, TexturePath):
        img = QImage(TexturePath)
        img = QGLWidget.convertToGLFormat(img)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits().asstring(img.byteCount()))


    def applyTarTexture(self, TarTexture):
        img = QImage("TarTexture.png")
        img = QGLWidget.convertToGLFormat(img)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits().asstring(img.byteCount()))

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def wheelEvent(self, event):
        numDegrees = event.angleDelta() / 8
        orientation = numDegrees.y()
        if orientation > 0:
            self.Tz -= 0.1  # zoom out
        else:
            self.Tz += 0.1  # zoom in
        self.updateGL()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()
        if event.buttons() & Qt.LeftButton:  # holding left button of mouse and moving will rotate the object
            self.setXRotation(self.xRot + 8 * dy)
            self.setYRotation(self.yRot + 8 * dx)
            self.LeftXRot = self.xRot + 8 * dy
            self.LeftYRot = self.yRot + 8 * dx
        elif event.buttons() & Qt.RightButton:  # holding right button of mouse and moving will translate the object
            self.Tx += dx / 100
            self.Ty -= dy / 100
            self.updateGL()
        elif event.buttons() & Qt.MidButton:  # holding middle button of mouse and moving will reset zoom/translations
            self.Tx = Ty = 0
            self.Tz = 1
            self.setXRotation(0)
            self.setYRotation(90)
            self.updateGL()

        self.lastPos = event.pos()

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 50:
            return

        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        self.aspect = float(width) / float(height)
        gluPerspective(60.0, self.aspect, 0.01, 10000)
        glMatrixMode(GL_MODELVIEW)

    def setClearColor(self, c):
        gl.glClearColor(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setColor(self, c):
        gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.xRot = angle
            self.xRotationChanged.emit(angle)
            self.update()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.yRot = angle
            self.yRotationChanged.emit(angle)
            self.update()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.xRot:
            self.zRot = angle
            self.zRotationChanged.emit(angle)
            self.update()

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
