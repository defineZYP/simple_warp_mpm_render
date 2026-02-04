import os
import numpy as np
import torch

def look_at_to_c2w(camera_position, target_position=[0.0, 0.0, 0.0], up_dir=[0.0, 0.0, 1.0]) -> np.ndarray:
    """
    Look at transform matrix

    :param camera_position: camera position
    :param target_position: target position, default is origin
    :param up_dir: up vector, default is z-axis up
    :return: camera to world matrix
    """

    camera_direction = np.array(camera_position) - np.array(target_position)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    camera_right = np.cross(np.array(up_dir), camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0
    translation_transform = np.eye(4)
    translation_transform[:3, -1] = -np.array(camera_position)
    look_at_transform = np.matmul(rotation_transform, translation_transform)
    return np.linalg.inv(look_at_transform)

