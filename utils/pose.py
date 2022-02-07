from gzip import READ
from tkinter import LEFT
import cv2
import numpy as np
import os


class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""
    def __init__(self, img_size=(480, 640)):
        self.size = img_size
        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]], [0, 0, 1]],
            dtype="double")
        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))
        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383],
                               [-2053.03596872]])
        if os.path.isfile('./assets/calibration.yml'):
            mtx, dist, rvecs, tvecs = self.load_coefficients(
                './assets/calibration.yml')
            self.camera_matrix = mtx
            self.dist_coeefs = dist
        #     self.r_vec = rvecs
        #     self.t_vec = tvecs
        # 3D model points.
        self.model_points_68 = self._get_full_model_points()

    def load_coefficients(self, path):
        '''Loads camera matrix and distortion coefficients.'''
        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        camera_matrix = cv_file.getNode('K').mat()
        dist_matrix = cv_file.getNode('D').mat()
        rvecs = cv_file.getNode('R').mat()
        tvecs = cv_file.getNode('T').mat()
        cv_file.release()
        return [camera_matrix, dist_matrix, rvecs, tvecs]

    def _get_full_model_points(self, filename='./assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)

        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1
        # fetch lnmk25
        lnmks_68 = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47, 27, 28, 29, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52,
            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67
        ]
        # correct to our facial landmarks
        model_points = model_points[lnmks_68]
        lnmk_scheme = [
            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 48, 50, 51,
            52, 54, 56, 57, 58
        ]
        # -------
        model_points = model_points[lnmk_scheme]
        # LE = np.mean(model_points[:6], axis=0, keepdims=True)
        # RE = np.mean(model_points[6:12], axis=0, keepdims=True)
        # N = model_points[13:14]  #13
        # LM = model_points[14:15]
        # RM = model_points[18:19]
        # model_points = np.concatenate([LE, RE, N, LM, RM], axis=0)
        #-------

        model_points = np.asarray(model_points)
        return model_points

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """
        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix,
                self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector
        (_, rotation_vector,
         translation_vector) = cv2.solvePnP(self.model_points_68,
                                            image_points,
                                            self.camera_matrix,
                                            self.dist_coeefs,
                                            rvec=self.r_vec,
                                            tvec=self.t_vec,
                                            useExtrinsicGuess=True)
        return (rotation_vector, translation_vector)