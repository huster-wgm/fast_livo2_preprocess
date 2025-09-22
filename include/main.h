 
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include<mutex>
#include <iostream> 
#include <sstream>
#include <unordered_map>
#include <opencv2/core/affine.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include "livox_ros_driver2/CustomMsg.h"
    double g;
    std::string topic_imu_in;
    std::string topic_imu_out;

    std::vector<std::string> topic_camera_in;
    std::vector<std::string> topic_camera_out;
    std::vector<std::string> topic_camera_undistort_out;    

    std::string topic_lidar_in;
    std::string topic_lidar_out;
    std::string topic_lidar_time_align;

    ros::Subscriber sub_lidar;
    ros::Publisher pub_lidar;
    ros::Publisher pub_lidar_time_align;
    
    ros::Subscriber sub_camera_0;
    ros::Subscriber sub_camera_1;

    ros::Publisher pub_camera_0;
    ros::Publisher pub_camera_undistort_0;

    ros::Publisher pub_camera_1;
    ros::Publisher pub_camera_undistort_1;

    ros::Subscriber sub_imu;
    ros::Publisher pub_imu;

    std::vector<std::vector<double>> distortion_coeffs;
    std::vector<std::vector<double>> intrinsics;
    std::vector<std::vector<int>> resolution;    
    std::vector<std::vector<int>> out_resolution;    

    std::vector<std::vector<cv::Mat>> undistor_map;    


void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);
void imageCallback_0(const sensor_msgs::CompressedImage::ConstPtr& msg);
void imageCallback_1(const sensor_msgs::CompressedImage::ConstPtr& msg);

namespace edu_ros {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    double time;
    uint8_t  tag;
    uint8_t  line;      
    double timestamp;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}
POINT_CLOUD_REGISTER_POINT_STRUCT(edu_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                      (uint8_t, tag, tag)
                                      (uint8_t, line, line)                                      
                                      (double, timestamp, timestamp)
)
