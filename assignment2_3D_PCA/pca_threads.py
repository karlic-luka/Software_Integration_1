from PyQt5.QtCore import QObject, QThread, pyqtSignal
import os
from imageio.v2 import imread
import numpy as np
from OBJ import OBJFastV
from scipy import linalg

class TextureThreadClass(QThread):

    finished = pyqtSignal(dict)
    running = False

    def __init__(self, parent=None):
        super(TextureThreadClass, self).__init__(parent)
        self.running = True

    def run(self):
        if self.running:
            self.process_PCA_texture()

    def stop(self):
        self.running = False
        print(f'Stopped thread for texture PCA')

    def process_PCA_texture(self):
        print(f'Running thread for texture PCA')
        texture_pca_result = {}
        try:
            texture_data, texture_mu = self.read_and_prepare_texture_models()
            eigenvectors, _, _ = linalg.svd(texture_data.transpose(), full_matrices=False)
            texture_weights = np.dot(texture_data, eigenvectors)
            texture_pca_result['eigenvectors_transposed_flattened'] = eigenvectors.transpose().flatten()
            texture_pca_result['mean'] = texture_mu
            texture_pca_result['weights'] = texture_weights
        except Exception as e:
            print(f'Error in thread for texture PCA: {e}')
        self.finished.emit(texture_pca_result)
        print(f'Finished thread for texture PCA')
        return

    def read_and_prepare_texture_models(self, dir=None):
        # Read the 2 models
        print(f'Inside thread for texture PCA')
        print(f'Reading texture models')
        if dir is None:
            dir = os.path.join(os.getcwd(), 'assignment2_3D_PCA', 'Database', 'Texture')
        textures_pngs = [file for file in os.listdir(dir) if file.endswith('.png')]
        textures_pngs = [os.path.join(dir, file) for file in textures_pngs]
        
        num_models = len(textures_pngs)
        sample_model = imread(textures_pngs[0])
        model_size = sample_model.shape
        print(f'Number of texture models: {num_models}')

        data = np.zeros((num_models, model_size[0] * model_size[1] * model_size[2]), dtype=np.float32)
        for i in range(0, num_models):
            data[i, :] = np.float32(imread(textures_pngs[i]) / 255).flatten()
            
        # Calculate the mean
        mu = np.mean(data, 0)
        data -= mu
        print(f'Finished reading texture models')
        return data, mu
    


class GeometryThreadClass(QThread):

    finished = pyqtSignal(dict)
    running = False

    def __init__(self, parent=None):
        super(GeometryThreadClass, self).__init__(parent)
        self.running = True
        return
    
    def run(self):
        if self.running:
            self.process_PCA_geometry()
        return
    
    def stop(self):
        self.running = False
        print(f'Stopped thread for geometry PCA')
        return

    def process_PCA_geometry(self):
        print(f'Running thread for geometry PCA')
        try:
            geometry_data, geometry_mu = self.read_and_prepare_geometry_models()
            eigenvectors, _, _ = linalg.svd(geometry_data.transpose(), full_matrices=False)
            geometry_weights = np.dot(geometry_data, eigenvectors)
            geometry_pca_result = {}
            geometry_pca_result['eigenvectors_transposed'] = eigenvectors.transpose()
            geometry_pca_result['mean'] = geometry_mu
            geometry_pca_result['weights'] = geometry_weights
        except Exception as e:
            print(f'Error in thread for geometry PCA: {e}')
        self.finished.emit(geometry_pca_result)
        print(f'Finished thread for geometry PCA')
        return
    
    def read_and_prepare_geometry_models(self, dir=None):
        print(f'Inside thread for geometry PCA')
        if dir is None:
            dir = os.path.join(os.getcwd(), 'assignment2_3D_PCA', 'Database', 'Geometry')
        geometry_obj_files = [file for file in os.listdir(dir) if file.endswith('.obj')]
        geometry_obj_files = [os.path.join(dir, file) for file in geometry_obj_files]

        num_models = len(geometry_obj_files)
        sample_model = OBJFastV(geometry_obj_files[0])
        num_vertices = len(sample_model.vertices)
        print(f'Number of geometry models: {num_models}')

        data = np.zeros((num_models, num_vertices * 3), dtype=np.float32) # 3 for x, y, z
        for i in range(0, num_models):
            vertices = OBJFastV(geometry_obj_files[i]).vertices
            x_coords = [row[0] for row in vertices]
            y_coords = [row[1] for row in vertices]
            z_coords = [row[2] for row in vertices]
            data[i, :] = np.hstack((x_coords, y_coords, z_coords))
        mu = np.mean(data, 0)
        data -= mu
        print(f'Finished reading geometry models')
        return data, mu