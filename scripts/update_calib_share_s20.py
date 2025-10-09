import os
import sys
import yaml
import numpy as np


class Intrinsic:
    def __init__(self, model: str, fx: float, fy: float, cx: float, cy: float, dist: list):
        self.model = model
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dist = dist

    def to_matrix(self):
        return [
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ]

    def resize(self, scale: float):
        self.fx *= scale
        self.fy *= scale
        self.cx *= scale
        self.cy *= scale


class Extrinsic:
    def __init__(self, transform_matrix: np.array):
        # Extract rotation (3x3) and translation (3x1) from 4x4 transformation matrix
        self.rotation = transform_matrix[:3, :3]  # 3x3 rotation matrix
        self.translation = transform_matrix[:3, 3].reshape(
            3, 1)  # 3x1 translation vector

    def to_matrix(self):
        """generate 4x4 transformation matrix from rotation and translation
        """
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = self.rotation
        extrinsic_matrix[:3, 3] = self.translation.flatten()
        return extrinsic_matrix


# 从share_s20的YAML文件解析标定参数
def parse_calib_yaml(yaml_file_path):
    import cv2

    # 使用cv2.FileStorage加载OpenCV风格的YAML文件
    fs = cv2.FileStorage(yaml_file_path, cv2.FILE_STORAGE_READ)

    cameras = {}

    # 先获取intrinsic节点
    intrinsic_node = fs.getNode("intrinsic")
    if not intrinsic_node.isNone():
        # 解析left相机
        fisheye_left = intrinsic_node.getNode("fisheye_left")
        if not fisheye_left.isNone():
            model_type = fisheye_left.getNode("model_type").string()
            camera_name = fisheye_left.getNode("camera_name").string()
            image_width = int(fisheye_left.getNode("image_width").real())
            image_height = int(fisheye_left.getNode("image_height").real())

            # 获取投影参数
            proj_params = fisheye_left.getNode("projection_parameters")
            fx = proj_params.getNode("A11").real()
            fy = proj_params.getNode("A22").real()
            cx = proj_params.getNode("u0").real()
            cy = proj_params.getNode("v0").real()

            # 获取畸变参数
            dist = [
                proj_params.getNode("k2").real(),
                proj_params.getNode("k3").real(),
                proj_params.getNode("k4").real(),
                proj_params.getNode("k5").real(),
                proj_params.getNode("k6").real(),
                proj_params.getNode("k7").real()
            ]

            intrinsic = Intrinsic(
                model=model_type,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                dist=dist
            )

            # 解析外参
            extrinsic_node = fs.getNode("extrinsic")
            lidar_leftcamera = extrinsic_node.getNode("lidar_leftcamera")
            transform_matrix = lidar_leftcamera.mat()
            extrinsic = Extrinsic(transform_matrix)

            cameras['left'] = {
                'width': image_width,
                'height': image_height,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic
            }

        # 解析right相机
        fisheye_right = intrinsic_node.getNode("fisheye_right")
        if not fisheye_right.isNone():
            model_type = fisheye_right.getNode("model_type").string()
            camera_name = fisheye_right.getNode("camera_name").string()
            image_width = int(fisheye_right.getNode("image_width").real())
            image_height = int(fisheye_right.getNode("image_height").real())

            # 获取投影参数
            proj_params = fisheye_right.getNode("projection_parameters")
            fx = proj_params.getNode("A11").real()
            fy = proj_params.getNode("A22").real()
            cx = proj_params.getNode("u0").real()
            cy = proj_params.getNode("v0").real()

            # 获取畸变参数
            dist = [
                proj_params.getNode("k2").real(),
                proj_params.getNode("k3").real(),
                proj_params.getNode("k4").real(),
                proj_params.getNode("k5").real(),
                proj_params.getNode("k6").real(),
                proj_params.getNode("k7").real()
            ]

            intrinsic = Intrinsic(
                model=model_type,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                dist=dist
            )

            # 解析外参
            lidar_rightcamera = extrinsic_node.getNode("lidar_rightcamera")
            transform_matrix = lidar_rightcamera.mat()
            extrinsic = Extrinsic(transform_matrix)

            cameras['right'] = {
                'width': image_width,
                'height': image_height,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic
            }

        # 解析middle相机
        fisheye_middle = intrinsic_node.getNode("fisheye_middle")
        if not fisheye_middle.isNone():
            image_width = int(fisheye_middle.getNode("image_width").real())
            image_height = int(fisheye_middle.getNode("image_height").real())

            # 获取相机矩阵
            camera_matrix = fisheye_middle.getNode("camera_matrix").mat()
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]

            # 获取畸变系数
            distortion_coeffs = fisheye_middle.getNode(
                "distortion_coefficients").mat()
            dist = distortion_coeffs.flatten().tolist()

            intrinsic = Intrinsic(
                model='Pinhole',
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                dist=dist
            )

            # 解析外参
            lidar_middlecamera = extrinsic_node.getNode("lidar_middlecamera")
            transform_matrix = lidar_middlecamera.mat()
            extrinsic = Extrinsic(transform_matrix)

            cameras['middle'] = {
                'width': image_width,
                'height': image_height,
                'intrinsic': intrinsic,
                'extrinsic': extrinsic
            }

    # 关闭FileStorage
    fs.release()

    return cameras


# 更新相机YAML配置
def update_camera_yaml(camera: dict, yaml_file_path: str):
    # Update the YAML data with new calibration info
    intrinsic = camera["intrinsic"]
    scale = 0.25  # Scale factor for resizing

    # Create ordered dictionary to maintain the exact order
    yaml_data = {
        "cam_model": "Pinhole",
        "scale": 1.0,
        "cam_width": int(camera["width"] * scale),
        "cam_height": int(camera["height"] * scale),
        "cam_fx": intrinsic.fx * scale,
        "cam_fy": intrinsic.fy * scale,
        "cam_cx": intrinsic.cx * scale,
        "cam_cy": intrinsic.cy * scale,
        "cam_d0": 0,
        "cam_d1": 0,
        "cam_d2": 0,
        "cam_d3": 0
    }

    # Write the updated calibration data to the YAML file with preserved order
    with open(yaml_file_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"Updated calibration data written to {yaml_file_path}")


# 更新FAST-LIVO2配置YAML
def update_fastlivo2_config_yaml(camera: dict, yaml_file_path: str):
    # Load existing YAML data
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Ensure extrin_calib exists
    if 'extrin_calib' not in yaml_data:
        yaml_data['extrin_calib'] = {}

    # Custom write function to maintain formatting
    def write_yaml(data, file):
        def format_list(lst):
            return "[" + ", ".join(f"{x:.7g}" for x in lst) + "]"

        # Write each section
        for key, value in data.items():
            if isinstance(value, dict):
                file.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        file.write(f"  {subkey}: {format_list(subvalue)}\n")
                    else:
                        file.write(f"  {subkey}: {subvalue}\n")
            else:
                file.write(f"{key}: {value}\n")

    # Update values while maintaining the list format
    yaml_data['extrin_calib']['Rcl'] = camera['extrinsic'].rotation.flatten().tolist()
    yaml_data['extrin_calib']['Pcl'] = camera['extrinsic'].translation.flatten().tolist()

    # Update img_topic
    tname = 'left' if use_left_cam else 'right'
    yaml_data['common']['img_topic'] = f"/camera/{tname}/jpeg_1k/undistort"

    # Write the updated YAML data back to the file
    with open(yaml_file_path, "w") as f:
        write_yaml(yaml_data, f)


# 更新数据预处理配置YAML
def update_preprocess_config_yaml(cameras: dict, yaml_file_path: str):
    # Load existing YAML data
    with open(yaml_file_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Update camera parameters for left and right cameras
    camera_mapping = {'left': 'cam0', 'right': 'cam1'}

    # Update refer_cam_id based on global use_left_cam
    yaml_data['refer_cam_id'] = 0 if use_left_cam else 1

    for cam_name, yaml_cam_key in camera_mapping.items():
        if cam_name in cameras:
            cam = cameras[cam_name]
            intrinsic = cam['intrinsic']
            extrinsic = cam['extrinsic']

            # Update T_lidar_cam as a flattened list
            yaml_data[yaml_cam_key]['T_lidar_cam'] = extrinsic.to_matrix(
            ).flatten().tolist()

            # Update intrinsics [fu, fv, cu, cv]
            yaml_data[yaml_cam_key]['intrinsics'] = [
                intrinsic.fx, intrinsic.fy,
                intrinsic.cx, intrinsic.cy
            ]

            # Update distortion coefficients
            yaml_data[yaml_cam_key]['distortion_coeffs'] = intrinsic.dist

            # Update resolution
            yaml_data[yaml_cam_key]['in_resolution'] = [
                int(cam['width']), int(cam['height'])
            ]

            yaml_data[yaml_cam_key]['out_resolution'] = [
                int(cam['width'] * 0.25), int(cam['height'] * 0.25)
            ]

            # update distortion model
            yaml_data[yaml_cam_key]['distortion_model'] = "polynomial"

            # update rostopic
            tname = 'left' if cam_name == 'left' else 'right'
            yaml_data[yaml_cam_key]['rostopic'] = f"/camera_agent/img_{tname}/compressed"

    # Custom write function to maintain formatting
    def write_yaml(data, file):
        def format_matrix(mat):
            if len(mat) == 16:  # 4x4 matrix
                rows = [mat[i:i+4] for i in range(0, 16, 4)]
                formatted = []
                for row in rows:
                    formatted.append(", ".join(f"{x:.16g}" for x in row))
                return "[\n    " + ",\n    ".join(formatted) + "]"
            else:  # Regular list
                return "[" + ", ".join(f"{x:.16g}" for x in mat) + "]"

        def write_section(section, indent=""):
            for key, value in section.items():
                if isinstance(value, dict):
                    file.write(f"{indent}{key}:\n")
                    write_section(value, indent + "  ")
                elif isinstance(value, list):
                    if key in ['T_lidar_cam', 'T_imu_lidar']:
                        file.write(f"{indent}{key}: {format_matrix(value)}\n")
                    else:
                        file.write(f"{indent}{key}: {format_matrix(value)}\n")
                else:
                    file.write(f"{indent}{key}: {value}\n")

        write_section(data)

    # Write the updated YAML data back to the file
    with open(yaml_file_path, "w") as f:
        write_yaml(yaml_data, f)


def update_gs_sdf_config_yaml(camera: dict, yaml_file_path: str, scale: float = 0.25):
    # First read the existing file to preserve the structure
    def opencv_matrix_constructor(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        rows = mapping["rows"]
        cols = mapping["cols"]
        data = mapping["data"]
        dt = mapping["dt"]
        data = np.array(data).reshape((rows, cols)).flatten().tolist()
        return {"rows": rows, "cols": cols, "dt": dt, "data": data}

    yaml.SafeLoader.add_constructor(
        "tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)

    with open(yaml_file_path, "r") as f:
        lines = f.readlines()

    if lines[0].startswith("%YAML:1.0"):
        lines = lines[1:]

    yaml_data = yaml.safe_load("".join(lines))

    # Update camera parameters
    intrinsic = camera['intrinsic']
    yaml_data['camera'].update({
        'model': 0,  # 0: pinhole
        'width': int(camera['width'] * scale),
        'height': int(camera['height'] * scale),
        'fx': intrinsic.fx * scale,
        'fy': intrinsic.fy * scale,
        'cx': intrinsic.cx * scale,
        'cy': intrinsic.cy * scale,
        'd0': 0.0,
        'd1': 0.0,
        'd2': 0.0,
        'd3': 0.0,
        'd4': 0.0  # GS-SDF uses 5 distortion parameters
    })

    # Convert extrinsic to 4x4 transformation matrix
    T_C_L = camera['extrinsic'].to_matrix()
    T_B_L = yaml_data['extrinsic'].get('T_B_L', {
        'rows': 4,
        'cols': 4,
        'dt': 'd',
        'data': [1.0, 0.0, 0.0, -0.011,
                 0.0, 1.0, 0.0, -0.02329,
                 0.0, 0.0, 1.0, 0.04412,
                 0.0, 0.0, 0.0, 1.0]
    })

    # Write the updated configuration
    with open(yaml_file_path, "w") as f:
        # Write header
        f.write("%YAML:1.0\n\n")

        # Write base configuration references if they exist
        if 'base_config' in yaml_data:
            f.write(f"base_config: \"{yaml_data['base_config']}\"\n")
        if 'scene_config' in yaml_data:
            f.write(f"scene_config: \"{yaml_data['scene_config']}\"\n")
        f.write("\n")

        # Write SDF parameters if they exist
        if 'sdf_iter_step' in yaml_data:
            f.write(f"sdf_iter_step: {yaml_data['sdf_iter_step']}\n")
        if 'gs_iter_step' in yaml_data:
            f.write(f"gs_iter_step: {yaml_data['gs_iter_step']}\n")
        f.write("\n")

        # Write other parameters if they exist
        if 'leaf_sizes' in yaml_data:
            f.write(f"leaf_sizes: {yaml_data['leaf_sizes']}\n")
        if 'bce_sigma' in yaml_data:
            f.write(
                f"bce_sigma: {yaml_data['bce_sigma']} # if get floaters, increase this value\n")
        f.write("\n")

        # Write camera section
        f.write("camera:\n")
        f.write(
            f"   model: {yaml_data['camera']['model']} # 0: pinhole; 1: equidistant\n\n")
        f.write(f"   width: {yaml_data['camera']['width']}\n")
        f.write(f"   height: {yaml_data['camera']['height']}\n\n")
        f.write(f"   fx: {yaml_data['camera']['fx']}\n")
        f.write(f"   fy: {yaml_data['camera']['fy']}\n")
        f.write(f"   cx: {yaml_data['camera']['cx']}\n")
        f.write(f"   cy: {yaml_data['camera']['cy']}\n\n")
        f.write(f"   d0: {yaml_data['camera']['d0']}\n")
        f.write(f"   d1: {yaml_data['camera']['d1']}\n")
        f.write(f"   d2: {yaml_data['camera']['d2']}\n")
        f.write(f"   d3: {yaml_data['camera']['d3']}\n")
        f.write(f"   d4: {yaml_data['camera']['d4']}\n\n")

        # Write extrinsic section
        f.write("extrinsic:\n")
        # Write T_C_L
        f.write("   # lidar to camera\n")
        f.write("   T_C_L: !!opencv-matrix\n")
        f.write("      rows: 4\n")
        f.write("      cols: 4\n")
        f.write("      dt: d\n")
        f.write("      data: [ " +
                ", ".join(f"{x:.16g}" for x in T_C_L[0]) + ",\n")
        f.write("              " +
                ", ".join(f"{x:.16g}" for x in T_C_L[1]) + ",\n")
        f.write("              " +
                ", ".join(f"{x:.16g}" for x in T_C_L[2]) + ",\n")
        f.write("              " +
                ", ".join(f"{x:.16g}" for x in T_C_L[3]) + " ]\n\n")

        # Write T_B_L
        f.write("   # lidar to base(imu)\n")
        f.write("   T_B_L: !!opencv-matrix\n")
        f.write("      rows: 4\n")
        f.write("      cols: 4\n")
        f.write("      dt: d\n")
        data = T_B_L['data']
        f.write(
            "      data: [ " + ", ".join(f"{data[i]:.16g}" for i in range(4)) + ", \n")
        f.write("              " +
                ", ".join(f"{data[i]:.16g}" for i in range(4, 8)) + ",\n")
        f.write("              " +
                ", ".join(f"{data[i]:.16g}" for i in range(8, 12)) + ",\n")
        f.write("              " +
                ", ".join(f"{data[i]:.16g}" for i in range(12, 16)) + " ]\n\n")

        # Write map section if it exists
        if 'map' in yaml_data:
            f.write("map:\n")
            f.write(f"   map_size: {yaml_data['map']['map_size']}\n")

    print(f"Updated GS-SDF config in {yaml_file_path}")


if __name__ == "__main__":
    # check if argv is empty
    if len(sys.argv) < 5:
        print("Usage: python3 update_calib_metacam_air.py <left/right> <fastlivo_ws> <gs_sdf_ws> <calib_file>")
        sys.exit(1)

    global use_left_cam
    use_left_cam = sys.argv[1] == "left"
    # fastlivo workspace, default is /home/ubuntu/fastlivo2_ws/src
    fastlivo_ws = sys.argv[2]
    # gs_sdf workspace, default is /home/ubuntu/gs_sdf_ws/src
    gs_sdf_ws = sys.argv[3]
    # calib_file, default is ./share_s20/outdoor_2025-09-13_10-54-072/info/calibration.yaml
    calib_file = sys.argv[4]
    cameras = parse_calib_yaml(calib_file)

    # update setting in edu_camera.yaml and edu.yaml
    camera = cameras['left'] if use_left_cam else cameras['right']
    update_camera_yaml(
        camera, f"{fastlivo_ws}/FAST-LIVO2/config/edu_camera.yaml")
    update_fastlivo2_config_yaml(
        camera, f"{fastlivo_ws}/FAST-LIVO2/config/edu.yaml")

    # update setting in data_preprocessing
    update_preprocess_config_yaml(
        cameras, f"{fastlivo_ws}/data_preprocessing/config/data_preprocessing.yaml")

    # update setting in gs_sdf
    update_gs_sdf_config_yaml(
        cameras["left"], f"{gs_sdf_ws}/GS-SDF/config/fast_livo/metacam_left.yaml")
    update_gs_sdf_config_yaml(
        cameras["right"], f"{gs_sdf_ws}/GS-SDF/config/fast_livo/metacam_right.yaml")
