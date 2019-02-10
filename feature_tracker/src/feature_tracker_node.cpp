#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "feature_tracker.h"
#include <iostream>
#include <algorithm>
#include <set>
#include <Eigen/Dense>
#include <random_numbers/random_numbers.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#define SHOW_UNDISTORTION 0
using namespace std;
vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;
using namespace std;
using namespace cv;
using namespace Eigen;



class ImageProcessor {
public:
  // Constructor
  ImageProcessor();

  // Destructor
  ~ImageProcessor();

  // Initialize the object.
  bool initialize();

  typedef boost::shared_ptr<ImageProcessor> Ptr;
  typedef boost::shared_ptr<const ImageProcessor> ConstPtr;



  /*
   * @brief ProcessorConfig Configuration parameters for
   *    feature detection and tracking.
   */
  struct ProcessorConfig {
    int grid_row;
    int grid_col;
    int grid_min_feature_num;
    int grid_max_feature_num;

    int pyramid_levels;
    int patch_size;
    int fast_threshold;
    int max_iteration;
    double track_precision;
    double ransac_threshold;
    double stereo_threshold;
  };

  /*
   * @brief FeatureIDType An alias for unsigned long long int.
   */
  typedef unsigned long long int FeatureIDType;

  /*
   * @brief FeatureMetaData Contains necessary information
   *    of a feature for easy access.
   */
  struct FeatureMetaData {
    FeatureIDType id;
    float response;
    int lifetime;
    cv::Point2f cam0_point;
    cv::Point2f cam1_point;
  };

  /*
   * @brief GridFeatures Organize features based on the grid
   *    they belong to. Note that the key is encoded by the
   *    grid index.
   */
  typedef std::map<int, std::vector<FeatureMetaData> > GridFeatures;

  /*
   * @brief keyPointCompareByResponse
   *    Compare two keypoints based on the response.
   */
  static bool keyPointCompareByResponse(
      const cv::KeyPoint& pt1,
      const cv::KeyPoint& pt2) {
    // Keypoint with higher response will be at the
    // beginning of the vector.
    return pt1.response > pt2.response;
  }
  /*
   * @brief featureCompareByResponse
   *    Compare two features based on the response.
   */
  static bool featureCompareByResponse(
      const FeatureMetaData& f1,
      const FeatureMetaData& f2) {
    // Features with higher response will be at the
    // beginning of the vector.
    return f1.response > f2.response;
  }
  /*
   * @brief featureCompareByLifetime
   *    Compare two features based on the lifetime.
   */
  static bool featureCompareByLifetime(
      const FeatureMetaData& f1,
      const FeatureMetaData& f2) {
    // Features with longer lifetime will be at the
    // beginning of the vector.
    return f1.lifetime > f2.lifetime;
  }

  /*
   * @brief loadParameters
   *    Load parameters from the parameter server.
   */
  bool loadParameters(ros::NodeHandle &n);

  /*
   * @brief createRosIO
   *    Create ros publisher and subscirbers.
   */
  bool createRosIO();

  /*
   * @brief stereoCallback
   *    Callback function for the stereo images.
   * @param cam0_img left image.
   * @param cam1_img right image.
   */
  void stereoCallback(
      const sensor_msgs::ImageConstPtr& cam0_img,
      const sensor_msgs::ImageConstPtr& cam1_img);

  /*
   * @brief imuCallback
   *    Callback function for the imu message.
   * @param msg IMU msg.
   */
  void imuCallback(const sensor_msgs::ImuConstPtr& msg);

  /*
   * @initializeFirstFrame
   *    Initialize the image processing sequence, which is
   *    bascially detect new features on the first set of
   *    stereo images.
   */
  void initializeFirstFrame();

  /*
   * @brief trackFeatures
   *    Tracker features on the newly received stereo images.
   */
  void trackFeatures();

  /*
   * @addNewFeatures
   *    Detect new features on the image to ensure that the
   *    features are uniformly distributed on the image.
   */
  void addNewFeatures();

  /*
   * @brief pruneGridFeatures
   *    Remove some of the features of a grid in case there are
   *    too many features inside of that grid, which ensures the
   *    number of features within each grid is bounded.
   */
  void pruneGridFeatures();

  /*
   * @brief publish
   *    Publish the features on the current image including
   *    both the tracked and newly detected ones.
   */
  void publish();

  /*
   * @brief drawFeaturesMono
   *    Draw tracked and newly detected features on the left
   *    image only.
   */
  void drawFeaturesMono();
  /*
   * @brief drawFeaturesStereo
   *    Draw tracked and newly detected features on the
   *    stereo images.
   */
  void drawFeaturesStereo();

  /*
   * @brief createImagePyramids
   *    Create image pyramids used for klt tracking.
   */
  void createImagePyramids();

  /*
   * @brief integrateImuData Integrates the IMU gyro readings
   *    between the two consecutive images, which is used for
   *    both tracking prediction and 2-point RANSAC.
   * @return cam0_R_p_c: a rotation matrix which takes a vector
   *    from previous cam0 frame to current cam0 frame.
   * @return cam1_R_p_c: a rotation matrix which takes a vector
   *    from previous cam1 frame to current cam1 frame.
   */
  void integrateImuData(cv::Matx33f& cam0_R_p_c,
      cv::Matx33f& cam1_R_p_c);

  /*
   * @brief predictFeatureTracking Compensates the rotation
   *    between consecutive camera frames so that feature
   *    tracking would be more robust and fast.
   * @param input_pts: features in the previous image to be tracked.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsic matrix of the camera.
   * @return compensated_pts: predicted locations of the features
   *    in the current image based on the provided rotation.
   *
   * Note that the input and output points are of pixel coordinates.
   */
  void predictFeatureTracking(
      const std::vector<cv::Point2f>& input_pts,
      const cv::Matx33f& R_p_c,
      const cv::Vec4d& intrinsics,
      std::vector<cv::Point2f>& compenstated_pts);

  /*
   * @brief twoPointRansac Applies two point ransac algorithm
   *    to mark the inliers in the input set.
   * @param pts1: first set of points.
   * @param pts2: second set of points.
   * @param R_p_c: a rotation matrix takes a vector in the previous
   *    camera frame to the current camera frame.
   * @param intrinsics: intrinsics of the camera.
   * @param distortion_model: distortion model of the camera.
   * @param distortion_coeffs: distortion coefficients.
   * @param inlier_error: acceptable error to be considered as an inlier.
   * @param success_probability: the required probability of success.
   * @return inlier_flag: 1 for inliers and 0 for outliers.
   */
  void twoPointRansac(
      const std::vector<cv::Point2f>& pts1,
      const std::vector<cv::Point2f>& pts2,
      const cv::Matx33f& R_p_c,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      const double& inlier_error,
      const double& success_probability,
      std::vector<int>& inlier_markers);
  void undistortPoints(
      const std::vector<cv::Point2f>& pts_in,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs,
      std::vector<cv::Point2f>& pts_out,
      const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
      const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));
  void rescalePoints(
      std::vector<cv::Point2f>& pts1,
      std::vector<cv::Point2f>& pts2,
      float& scaling_factor);
  std::vector<cv::Point2f> distortPoints(
      const std::vector<cv::Point2f>& pts_in,
      const cv::Vec4d& intrinsics,
      const std::string& distortion_model,
      const cv::Vec4d& distortion_coeffs);

  /*
   * @brief stereoMatch Matches features with stereo image pairs.
   * @param cam0_points: points in the primary image.
   * @return cam1_points: points in the secondary image.
   * @return inlier_markers: 1 if the match is valid, 0 otherwise.
   */
  void stereoMatch(
      const std::vector<cv::Point2f>& cam0_points,
      std::vector<cv::Point2f>& cam1_points,
      std::vector<unsigned char>& inlier_markers);

  /*
   * @brief removeUnmarkedElements Remove the unmarked elements
   *    within a vector.
   * @param raw_vec: vector with outliers.
   * @param markers: 0 will represent a outlier, 1 will be an inlier.
   * @return refined_vec: a vector without outliers.
   *
   * Note that the order of the inliers in the raw_vec is perserved
   * in the refined_vec.
   */
  template <typename T>
  void removeUnmarkedElements(
      const std::vector<T>& raw_vec,
      const std::vector<unsigned char>& markers,
      std::vector<T>& refined_vec) {
    if (raw_vec.size() != markers.size()) {
      ROS_WARN("The input size of raw_vec(%lu) and markers(%lu) does not match...",
          raw_vec.size(), markers.size());
    }
    for (int i = 0; i < markers.size(); ++i) {
      if (markers[i] == 0) continue;
      refined_vec.push_back(raw_vec[i]);
    }
    return;
  }

  // Indicate if this is the first image message.
  bool is_first_img;

  // ID for the next new feature.
  FeatureIDType next_feature_id;

  // Feature detector
  ProcessorConfig processor_config;
  cv::Ptr<cv::Feature2D> detector_ptr;

  // IMU message buffer.
  std::vector<sensor_msgs::Imu> imu_msg_buffer;

  // Camera calibration parameters
  std::string cam0_distortion_model;
  cv::Vec2i cam0_resolution;
  cv::Vec4d cam0_intrinsics;
  cv::Vec4d cam0_distortion_coeffs;

  std::string cam1_distortion_model;
  cv::Vec2i cam1_resolution;
  cv::Vec4d cam1_intrinsics;
  cv::Vec4d cam1_distortion_coeffs;

  // Take a vector from cam0 frame to the IMU frame.
  cv::Matx33d R_cam0_imu;
  cv::Vec3d t_cam0_imu;
  // Take a vector from cam1 frame to the IMU frame.
  cv::Matx33d R_cam1_imu;
  cv::Vec3d t_cam1_imu;

  // Previous and current images
  cv_bridge::CvImageConstPtr cam0_prev_img_ptr;
  cv_bridge::CvImageConstPtr cam0_curr_img_ptr;
  cv_bridge::CvImageConstPtr cam1_curr_img_ptr;

  // Pyramids for previous and current image
  std::vector<cv::Mat> prev_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam0_pyramid_;
  std::vector<cv::Mat> curr_cam1_pyramid_;

  // Features in the previous and current image.
  boost::shared_ptr<GridFeatures> prev_features_ptr;
  boost::shared_ptr<GridFeatures> curr_features_ptr;

  // Number of features after each outlier removal step.
  int before_tracking;
  int after_tracking;
  int after_matching;
  int after_ransac;

  // Debugging
  std::map<FeatureIDType, int> feature_lifetime;
  void updateFeatureLifetime();
  void featureLifetimeStatistics();
};

typedef ImageProcessor::Ptr ImageProcessorPtr;
typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;


ImageProcessor::ImageProcessor() :
  is_first_img(true),
  //img_transport(n),
  next_feature_id(0),
//  stereo_sub(10),
  prev_features_ptr(new GridFeatures()),
  curr_features_ptr(new GridFeatures()) {
  return;
}

ImageProcessor::~ImageProcessor() {
  destroyAllWindows();
  //ROS_INFO("Feature lifetime statistics:");
  //featureLifetimeStatistics();
  return;
}
template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}
bool ImageProcessor::loadParameters(ros::NodeHandle &n) {
  // Camera calibration parameters
  std::string config_file;
    config_file = readParam<std::string>(n, "config_file");
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["distortion_model"] >> cam0_distortion_model;
  
      cam1_distortion_model=cam0_distortion_model;
  
  cam0_resolution[0] = fsSettings["image_width"];
  cam0_resolution[1] = fsSettings["image_height"];
  
  cam1_resolution[0] = cam0_resolution[0];
  cam1_resolution[1] = cam0_resolution[1];

  cv::FileNode stereo_n = fsSettings["projection_parameters"];
  cam0_intrinsics[0] = static_cast<double>(stereo_n["fx"]);
  cam0_intrinsics[1] = static_cast<double>(stereo_n["fy"]);
  cam0_intrinsics[2] = static_cast<double>(stereo_n["cx"]);
  cam0_intrinsics[3] = static_cast<double>(stereo_n["cy"]);
  
  cam1_intrinsics[0] = cam0_intrinsics[0];
  cam1_intrinsics[1] = cam0_intrinsics[1];
  cam1_intrinsics[2] = cam0_intrinsics[2];
  cam1_intrinsics[3] = cam0_intrinsics[3];

  stereo_n = fsSettings["distortion_parameters"];
  cam0_distortion_coeffs[0] = static_cast<double>(stereo_n["k1"]);
  cam0_distortion_coeffs[1] = static_cast<double>(stereo_n["k2"]);
  cam0_distortion_coeffs[2] = static_cast<double>(stereo_n["p1"]);
  cam0_distortion_coeffs[3] = static_cast<double>(stereo_n["p2"]);

 
  cam1_distortion_coeffs[0] = cam0_distortion_coeffs[0];
  cam1_distortion_coeffs[1] = cam0_distortion_coeffs[1];
  cam1_distortion_coeffs[2] = cam0_distortion_coeffs[2];
  cam1_distortion_coeffs[3] = cam0_distortion_coeffs[3];
  
  cv::Mat R;
  fsSettings["extrinsicRotation"] >> R;
  cv::Mat T;
  fsSettings["extrinsicTranslation"] >> T;
  cv::Matx33d R_imu_cam0 = R;
  cv::Vec3d   t_imu_cam0 = T;
  R_cam0_imu = R_imu_cam0;//R_imu_cam0.t();
  t_cam0_imu = t_imu_cam0;//-R_imu_cam0.t() * t_imu_cam0;
  
  cv::Mat R1;
  fsSettings["extrinsicRotation_right"] >> R1;
  cv::Mat T1;
  fsSettings["extrinsicTranslation_right"] >> T1;
  cv::Matx33d R_imu_cam1 = R1;
  cv::Vec3d   t_imu_cam1 = T1;
  R_cam1_imu = R_imu_cam1;//R_imu_cam1.t();
  t_cam1_imu = t_imu_cam1;//-R_imu_cam1.t() * t_imu_cam1;
  // Processor parameters
  processor_config.grid_row = 6;
  processor_config.grid_col = 6;
  processor_config.grid_min_feature_num = 3;
  processor_config.grid_max_feature_num = 6;
  processor_config.pyramid_levels = 3;
  processor_config.patch_size = 15;
  processor_config.fast_threshold = 10;
  processor_config.max_iteration = 30;
  processor_config.track_precision = 0.01;
  processor_config.ransac_threshold = 3;
  processor_config.stereo_threshold = 5;
// Create feature detector.
  detector_ptr = FastFeatureDetector::create(
      processor_config.fast_threshold);
}
#if 0
bool ImageProcessor::createRosIO() {
  feature_pub = nh.advertise<CameraMeasurement>(
      "features", 3);
  tracking_info_pub = nh.advertise<TrackingInfo>(
      "tracking_info", 1);
  image_transport::ImageTransport it(nh);
  debug_stereo_pub = it.advertise("debug_stereo_image", 1);

  cam0_img_sub.subscribe(nh, "cam0_image", 10);
  cam1_img_sub.subscribe(nh, "cam1_image", 10);
  stereo_sub.connectInput(cam0_img_sub, cam1_img_sub);
  stereo_sub.registerCallback(&ImageProcessor::stereoCallback, this);
  imu_sub = nh.subscribe("imu", 50,
      &ImageProcessor::imuCallback, this);

  return true;
}

bool ImageProcessor::initialize() {
  if (!loadParameters()) return false;
  ROS_INFO("Finish loading ROS parameters...");

  // Create feature detector.
  detector_ptr = FastFeatureDetector::create(
      processor_config.fast_threshold);

  if (!createRosIO()) return false;
  ROS_INFO("Finish creating ROS IO...");

  return true;
}
#endif
void ImageProcessor::stereoCallback(
    const sensor_msgs::ImageConstPtr& cam0_img,
    const sensor_msgs::ImageConstPtr& cam1_img) {

  //cout << "==================================" << endl;

  // Get the current image.
  cam0_curr_img_ptr = cv_bridge::toCvShare(cam0_img,
      sensor_msgs::image_encodings::MONO8);
  cam1_curr_img_ptr = cv_bridge::toCvShare(cam1_img,
      sensor_msgs::image_encodings::MONO8);

  // Build the image pyramids once since they're used at multiple places
  createImagePyramids();

  // Detect features in the first frame.
  if (is_first_img) {
    ros::Time start_time = ros::Time::now();
    initializeFirstFrame();
    //ROS_INFO("Detection time: %f",
    //    (ros::Time::now()-start_time).toSec());
    is_first_img = false;

    // Draw results.
    start_time = ros::Time::now();
    drawFeaturesStereo();
    //ROS_INFO("Draw features: %f",
    //    (ros::Time::now()-start_time).toSec());
  } else {
    // Track the feature in the previous image.
    ros::Time start_time = ros::Time::now();
    trackFeatures();
    //ROS_INFO("Tracking time: %f",
    //    (ros::Time::now()-start_time).toSec());

    // Add new features into the current image.
    start_time = ros::Time::now();
    addNewFeatures();
    //ROS_INFO("Addition time: %f",
    //    (ros::Time::now()-start_time).toSec());

    // Add new features into the current image.
    start_time = ros::Time::now();
    pruneGridFeatures();
    //ROS_INFO("Prune grid features: %f",
    //    (ros::Time::now()-start_time).toSec());

    // Draw results.
    start_time = ros::Time::now();
    drawFeaturesStereo();
    //ROS_INFO("Draw features: %f",
    //    (ros::Time::now()-start_time).toSec());
  }

  //ros::Time start_time = ros::Time::now();
  //updateFeatureLifetime();
  //ROS_INFO("Statistics: %f",
  //    (ros::Time::now()-start_time).toSec());

  // Publish features in the current image.
  ros::Time start_time = ros::Time::now();
  publish();
  //ROS_INFO("Publishing: %f",
  //    (ros::Time::now()-start_time).toSec());

  // Update the previous image and previous features.
  cam0_prev_img_ptr = cam0_curr_img_ptr;
  prev_features_ptr = curr_features_ptr;
  std::swap(prev_cam0_pyramid_, curr_cam0_pyramid_);

  // Initialize the current features to empty vectors.
  curr_features_ptr.reset(new GridFeatures());
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    (*curr_features_ptr)[code] = vector<FeatureMetaData>(0);
  }

  return;
}

void ImageProcessor::imuCallback(
    const sensor_msgs::ImuConstPtr& msg) {
  // Wait for the first image to be set.
  if (is_first_img) return;
  imu_msg_buffer.push_back(*msg);
  return;
}

void ImageProcessor::createImagePyramids() {
  const Mat& curr_cam0_img = cam0_curr_img_ptr->image;
  buildOpticalFlowPyramid(
      curr_cam0_img, curr_cam0_pyramid_,
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels, true, BORDER_REFLECT_101,
      BORDER_CONSTANT, false);

  const Mat& curr_cam1_img = cam1_curr_img_ptr->image;
  buildOpticalFlowPyramid(
      curr_cam1_img, curr_cam1_pyramid_,
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels, true, BORDER_REFLECT_101,
      BORDER_CONSTANT, false);
}

void ImageProcessor::initializeFirstFrame() {
  // Size of each grid.
  const Mat& img = cam0_curr_img_ptr->image;
  static int grid_height = img.rows / processor_config.grid_row;
  static int grid_width = img.cols / processor_config.grid_col;

  // Detect new features on the frist image.
  vector<KeyPoint> new_features(0);
  detector_ptr->detect(img, new_features);

  // Find the stereo matched points for the newly
  // detected features.
  vector<cv::Point2f> cam0_points(new_features.size());
  for (int i = 0; i < new_features.size(); ++i)
    cam0_points[i] = new_features[i].pt;

  vector<cv::Point2f> cam1_points(0);
  vector<unsigned char> inlier_markers(0);
  stereoMatch(cam0_points, cam1_points, inlier_markers);

  vector<cv::Point2f> cam0_inliers(0);
  vector<cv::Point2f> cam1_inliers(0);
  vector<float> response_inliers(0);
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cam0_inliers.push_back(cam0_points[i]);
    cam1_inliers.push_back(cam1_points[i]);
    response_inliers.push_back(new_features[i].response);
  }

  // Group the features into grids
  GridFeatures grid_new_features;
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code)
      grid_new_features[code] = vector<FeatureMetaData>(0);

  for (int i = 0; i < cam0_inliers.size(); ++i) {
    const cv::Point2f& cam0_point = cam0_inliers[i];
    const cv::Point2f& cam1_point = cam1_inliers[i];
    const float& response = response_inliers[i];

    int row = static_cast<int>(cam0_point.y / grid_height);
    int col = static_cast<int>(cam0_point.x / grid_width);
    int code = row*processor_config.grid_col + col;

    FeatureMetaData new_feature;
    new_feature.response = response;
    new_feature.cam0_point = cam0_point;
    new_feature.cam1_point = cam1_point;
    grid_new_features[code].push_back(new_feature);
  }

  // Sort the new features in each grid based on its response.
  for (auto& item : grid_new_features)
    std::sort(item.second.begin(), item.second.end(),
        &ImageProcessor::featureCompareByResponse);

  // Collect new features within each grid with high response.
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
    vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

    for (int k = 0; k < processor_config.grid_min_feature_num &&
        k < new_features_this_grid.size(); ++k) {
      features_this_grid.push_back(new_features_this_grid[k]);
      features_this_grid.back().id = next_feature_id++;
      features_this_grid.back().lifetime = 1;
    }
  }

  return;
}

void ImageProcessor::predictFeatureTracking(
    const vector<cv::Point2f>& input_pts,
    const cv::Matx33f& R_p_c,
    const cv::Vec4d& intrinsics,
    vector<cv::Point2f>& compensated_pts) {

  // Return directly if there are no input features.
  if (input_pts.size() == 0) {
    compensated_pts.clear();
    return;
  }
  compensated_pts.resize(input_pts.size());

  // Intrinsic matrix.
  cv::Matx33f K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);
  cv::Matx33f H = K * R_p_c * K.inv();

  for (int i = 0; i < input_pts.size(); ++i) {
    cv::Vec3f p1(input_pts[i].x, input_pts[i].y, 1.0f);
    cv::Vec3f p2 = H * p1;
    compensated_pts[i].x = p2[0] / p2[2];
    compensated_pts[i].y = p2[1] / p2[2];
  }

  return;
}

void ImageProcessor::trackFeatures() {
  // Size of each grid.
  static int grid_height =
    cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =
    cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  // Compute a rough relative rotation which takes a vector
  // from the previous frame to the current frame.
  Matx33f cam0_R_p_c;
  Matx33f cam1_R_p_c;
  integrateImuData(cam0_R_p_c, cam1_R_p_c);

  // Organize the features in the previous image.
  vector<FeatureIDType> prev_ids(0);
  vector<int> prev_lifetime(0);
  vector<Point2f> prev_cam0_points(0);
  vector<Point2f> prev_cam1_points(0);

  for (const auto& item : *prev_features_ptr) {
    for (const auto& prev_feature : item.second) {
      prev_ids.push_back(prev_feature.id);
      prev_lifetime.push_back(prev_feature.lifetime);
      prev_cam0_points.push_back(prev_feature.cam0_point);
      prev_cam1_points.push_back(prev_feature.cam1_point);
    }
  }

  // Number of the features before tracking.
  before_tracking = prev_cam0_points.size();

  // Abort tracking if there is no features in
  // the previous frame.
  if (prev_ids.size() == 0) return;

  // Track features using LK optical flow method.
  vector<Point2f> curr_cam0_points(0);
  vector<unsigned char> track_inliers(0);

  predictFeatureTracking(prev_cam0_points,
      cam0_R_p_c, cam0_intrinsics, curr_cam0_points);

  calcOpticalFlowPyrLK(
      prev_cam0_pyramid_, curr_cam0_pyramid_,
      prev_cam0_points, curr_cam0_points,
      track_inliers, noArray(),
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
        processor_config.max_iteration,
        processor_config.track_precision),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  // Mark those tracked points out of the image region
  // as untracked.
  for (int i = 0; i < curr_cam0_points.size(); ++i) {
    if (track_inliers[i] == 0) continue;
    if (curr_cam0_points[i].y < 0 ||
        curr_cam0_points[i].y > cam0_curr_img_ptr->image.rows-1 ||
        curr_cam0_points[i].x < 0 ||
        curr_cam0_points[i].x > cam0_curr_img_ptr->image.cols-1)
      track_inliers[i] = 0;
  }

  // Collect the tracked points.
  vector<FeatureIDType> prev_tracked_ids(0);
  vector<int> prev_tracked_lifetime(0);
  vector<Point2f> prev_tracked_cam0_points(0);
  vector<Point2f> prev_tracked_cam1_points(0);
  vector<Point2f> curr_tracked_cam0_points(0);

  removeUnmarkedElements(
      prev_ids, track_inliers, prev_tracked_ids);
  removeUnmarkedElements(
      prev_lifetime, track_inliers, prev_tracked_lifetime);
  removeUnmarkedElements(
      prev_cam0_points, track_inliers, prev_tracked_cam0_points);
  removeUnmarkedElements(
      prev_cam1_points, track_inliers, prev_tracked_cam1_points);
  removeUnmarkedElements(
      curr_cam0_points, track_inliers, curr_tracked_cam0_points);

  // Number of features left after tracking.
  after_tracking = curr_tracked_cam0_points.size();


  // Outlier removal involves three steps, which forms a close
  // loop between the previous and current frames of cam0 (left)
  // and cam1 (right). Assuming the stereo matching between the
  // previous cam0 and cam1 images are correct, the three steps are:
  //
  // prev frames cam0 ----------> cam1
  //              |                |
  //              |ransac          |ransac
  //              |   stereo match |
  // curr frames cam0 ----------> cam1
  //
  // 1) Stereo matching between current images of cam0 and cam1.
  // 2) RANSAC between previous and current images of cam0.
  // 3) RANSAC between previous and current images of cam1.
  //
  // For Step 3, tracking between the images is no longer needed.
  // The stereo matching results are directly used in the RANSAC.

  // Step 1: stereo matching.
  vector<Point2f> curr_cam1_points(0);
  vector<unsigned char> match_inliers(0);
  stereoMatch(curr_tracked_cam0_points, curr_cam1_points, match_inliers);

  vector<FeatureIDType> prev_matched_ids(0);
  vector<int> prev_matched_lifetime(0);
  vector<Point2f> prev_matched_cam0_points(0);
  vector<Point2f> prev_matched_cam1_points(0);
  vector<Point2f> curr_matched_cam0_points(0);
  vector<Point2f> curr_matched_cam1_points(0);

  removeUnmarkedElements(
      prev_tracked_ids, match_inliers, prev_matched_ids);
  removeUnmarkedElements(
      prev_tracked_lifetime, match_inliers, prev_matched_lifetime);
  removeUnmarkedElements(
      prev_tracked_cam0_points, match_inliers, prev_matched_cam0_points);
  removeUnmarkedElements(
      prev_tracked_cam1_points, match_inliers, prev_matched_cam1_points);
  removeUnmarkedElements(
      curr_tracked_cam0_points, match_inliers, curr_matched_cam0_points);
  removeUnmarkedElements(
      curr_cam1_points, match_inliers, curr_matched_cam1_points);

  // Number of features left after stereo matching.
  after_matching = curr_matched_cam0_points.size();

  // Step 2 and 3: RANSAC on temporal image pairs of cam0 and cam1.
  vector<int> cam0_ransac_inliers(0);
  twoPointRansac(prev_matched_cam0_points, curr_matched_cam0_points,
      cam0_R_p_c, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, processor_config.ransac_threshold,
      0.99, cam0_ransac_inliers);

  vector<int> cam1_ransac_inliers(0);
  twoPointRansac(prev_matched_cam1_points, curr_matched_cam1_points,
      cam1_R_p_c, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, processor_config.ransac_threshold,
      0.99, cam1_ransac_inliers);

  // Number of features after ransac.
  after_ransac = 0;

  for (int i = 0; i < cam0_ransac_inliers.size(); ++i) {
    if (cam0_ransac_inliers[i] == 0 ||
        cam1_ransac_inliers[i] == 0) continue;
    int row = static_cast<int>(
        curr_matched_cam0_points[i].y / grid_height);
    int col = static_cast<int>(
        curr_matched_cam0_points[i].x / grid_width);
    int code = row*processor_config.grid_col + col;
    (*curr_features_ptr)[code].push_back(FeatureMetaData());

    FeatureMetaData& grid_new_feature = (*curr_features_ptr)[code].back();
    grid_new_feature.id = prev_matched_ids[i];
    grid_new_feature.lifetime = ++prev_matched_lifetime[i];
    grid_new_feature.cam0_point = curr_matched_cam0_points[i];
    grid_new_feature.cam1_point = curr_matched_cam1_points[i];

    ++after_ransac;
  }

  // Compute the tracking rate.
  int prev_feature_num = 0;
  for (const auto& item : *prev_features_ptr)
    prev_feature_num += item.second.size();

  int curr_feature_num = 0;
  for (const auto& item : *curr_features_ptr)
    curr_feature_num += item.second.size();

  ROS_INFO_THROTTLE(0.5,
      "\033[0;32m candidates: %d; track: %d; match: %d; ransac: %d/%d=%f\033[0m",
      before_tracking, after_tracking, after_matching,
      curr_feature_num, prev_feature_num,
      static_cast<double>(curr_feature_num)/
      (static_cast<double>(prev_feature_num)+1e-5));
  //printf(
  //    "\033[0;32m candidates: %d; raw track: %d; stereo match: %d; ransac: %d/%d=%f\033[0m\n",
  //    before_tracking, after_tracking, after_matching,
  //    curr_feature_num, prev_feature_num,
  //    static_cast<double>(curr_feature_num)/
  //    (static_cast<double>(prev_feature_num)+1e-5));

  return;
}

void ImageProcessor::stereoMatch(
    const vector<cv::Point2f>& cam0_points,
    vector<cv::Point2f>& cam1_points,
    vector<unsigned char>& inlier_markers) {

  if (cam0_points.size() == 0) return;

  if(cam1_points.size() == 0) {
    // Initialize cam1_points by projecting cam0_points to cam1 using the
    // rotation from stereo extrinsics
    const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
    vector<cv::Point2f> cam0_points_undistorted;
    undistortPoints(cam0_points, cam0_intrinsics, cam0_distortion_model,
                    cam0_distortion_coeffs, cam0_points_undistorted,
                    R_cam0_cam1);
    cam1_points = distortPoints(cam0_points_undistorted, cam1_intrinsics,
                                cam1_distortion_model, cam1_distortion_coeffs);
  }

  // Track features using LK optical flow method.
  calcOpticalFlowPyrLK(curr_cam0_pyramid_, curr_cam1_pyramid_,
      cam0_points, cam1_points,
      inlier_markers, noArray(),
      Size(processor_config.patch_size, processor_config.patch_size),
      processor_config.pyramid_levels,
      TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                   processor_config.max_iteration,
                   processor_config.track_precision),
      cv::OPTFLOW_USE_INITIAL_FLOW);

  // Mark those tracked points out of the image region
  // as untracked.
  for (int i = 0; i < cam1_points.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    if (cam1_points[i].y < 0 ||
        cam1_points[i].y > cam1_curr_img_ptr->image.rows-1 ||
        cam1_points[i].x < 0 ||
        cam1_points[i].x > cam1_curr_img_ptr->image.cols-1)
      inlier_markers[i] = 0;
  }

  // Compute the relative rotation between the cam0
  // frame and cam1 frame.
  const cv::Matx33d R_cam0_cam1 = R_cam1_imu.t() * R_cam0_imu;
  const cv::Vec3d t_cam0_cam1 = R_cam1_imu.t() * (t_cam0_imu-t_cam1_imu);
  // Compute the essential matrix.
  const cv::Matx33d t_cam0_cam1_hat(
      0.0, -t_cam0_cam1[2], t_cam0_cam1[1],
      t_cam0_cam1[2], 0.0, -t_cam0_cam1[0],
      -t_cam0_cam1[1], t_cam0_cam1[0], 0.0);
  const cv::Matx33d E = t_cam0_cam1_hat * R_cam0_cam1;

  // Further remove outliers based on the known
  // essential matrix.
  vector<cv::Point2f> cam0_points_undistorted(0);
  vector<cv::Point2f> cam1_points_undistorted(0);
  undistortPoints(
      cam0_points, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, cam0_points_undistorted);
  undistortPoints(
      cam1_points, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, cam1_points_undistorted);

  double norm_pixel_unit = 4.0 / (
      cam0_intrinsics[0]+cam0_intrinsics[1]+
      cam1_intrinsics[0]+cam1_intrinsics[1]);

  for (int i = 0; i < cam0_points_undistorted.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cv::Vec3d pt0(cam0_points_undistorted[i].x,
        cam0_points_undistorted[i].y, 1.0);
    cv::Vec3d pt1(cam1_points_undistorted[i].x,
        cam1_points_undistorted[i].y, 1.0);
    cv::Vec3d epipolar_line = E * pt0;
    double error = fabs((pt1.t() * epipolar_line)[0]) / sqrt(
        epipolar_line[0]*epipolar_line[0]+
        epipolar_line[1]*epipolar_line[1]);
    if (error > processor_config.stereo_threshold*norm_pixel_unit)
      inlier_markers[i] = 0;
  }

  return;
}

void ImageProcessor::addNewFeatures() {
  const Mat& curr_img = cam0_curr_img_ptr->image;

  // Size of each grid.
  static int grid_height =
    cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =
    cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  // Create a mask to avoid redetecting existing features.
  Mat mask(curr_img.rows, curr_img.cols, CV_8U, Scalar(1));

  for (const auto& features : *curr_features_ptr) {
    for (const auto& feature : features.second) {
      const int y = static_cast<int>(feature.cam0_point.y);
      const int x = static_cast<int>(feature.cam0_point.x);

      int up_lim = y-2, bottom_lim = y+3,
          left_lim = x-2, right_lim = x+3;
      if (up_lim < 0) up_lim = 0;
      if (bottom_lim > curr_img.rows) bottom_lim = curr_img.rows;
      if (left_lim < 0) left_lim = 0;
      if (right_lim > curr_img.cols) right_lim = curr_img.cols;

      Range row_range(up_lim, bottom_lim);
      Range col_range(left_lim, right_lim);
      mask(row_range, col_range) = 0;
    }
  }

  // Detect new features.
  vector<KeyPoint> new_features(0);
  detector_ptr->detect(curr_img, new_features, mask);

  // Collect the new detected features based on the grid.
  // Select the ones with top response within each grid afterwards.
  vector<vector<KeyPoint> > new_feature_sieve(
      processor_config.grid_row*processor_config.grid_col);
  for (const auto& feature : new_features) {
    int row = static_cast<int>(feature.pt.y / grid_height);
    int col = static_cast<int>(feature.pt.x / grid_width);
    new_feature_sieve[
      row*processor_config.grid_col+col].push_back(feature);
  }

  new_features.clear();
  for (auto& item : new_feature_sieve) {
    if (item.size() > processor_config.grid_max_feature_num) {
      std::sort(item.begin(), item.end(),
          &ImageProcessor::keyPointCompareByResponse);
      item.erase(
          item.begin()+processor_config.grid_max_feature_num, item.end());
    }
    new_features.insert(new_features.end(), item.begin(), item.end());
  }

  int detected_new_features = new_features.size();

  // Find the stereo matched points for the newly
  // detected features.
  vector<cv::Point2f> cam0_points(new_features.size());
  for (int i = 0; i < new_features.size(); ++i)
    cam0_points[i] = new_features[i].pt;

  vector<cv::Point2f> cam1_points(0);
  vector<unsigned char> inlier_markers(0);
  stereoMatch(cam0_points, cam1_points, inlier_markers);

  vector<cv::Point2f> cam0_inliers(0);
  vector<cv::Point2f> cam1_inliers(0);
  vector<float> response_inliers(0);
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] == 0) continue;
    cam0_inliers.push_back(cam0_points[i]);
    cam1_inliers.push_back(cam1_points[i]);
    response_inliers.push_back(new_features[i].response);
  }

  int matched_new_features = cam0_inliers.size();

  if (matched_new_features < 5 &&
      static_cast<double>(matched_new_features)/
      static_cast<double>(detected_new_features) < 0.1)
    ROS_WARN("Images at [%f] seems unsynced...",
        cam0_curr_img_ptr->header.stamp.toSec());

  // Group the features into grids
  GridFeatures grid_new_features;
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code)
      grid_new_features[code] = vector<FeatureMetaData>(0);

  for (int i = 0; i < cam0_inliers.size(); ++i) {
    const cv::Point2f& cam0_point = cam0_inliers[i];
    const cv::Point2f& cam1_point = cam1_inliers[i];
    const float& response = response_inliers[i];

    int row = static_cast<int>(cam0_point.y / grid_height);
    int col = static_cast<int>(cam0_point.x / grid_width);
    int code = row*processor_config.grid_col + col;

    FeatureMetaData new_feature;
    new_feature.response = response;
    new_feature.cam0_point = cam0_point;
    new_feature.cam1_point = cam1_point;
    grid_new_features[code].push_back(new_feature);
  }

  // Sort the new features in each grid based on its response.
  for (auto& item : grid_new_features)
    std::sort(item.second.begin(), item.second.end(),
        &ImageProcessor::featureCompareByResponse);

  int new_added_feature_num = 0;
  // Collect new features within each grid with high response.
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features_this_grid = (*curr_features_ptr)[code];
    vector<FeatureMetaData>& new_features_this_grid = grid_new_features[code];

    if (features_this_grid.size() >=
        processor_config.grid_min_feature_num) continue;

    int vacancy_num = processor_config.grid_min_feature_num -
      features_this_grid.size();
    for (int k = 0;
        k < vacancy_num && k < new_features_this_grid.size(); ++k) {
      features_this_grid.push_back(new_features_this_grid[k]);
      features_this_grid.back().id = next_feature_id++;
      features_this_grid.back().lifetime = 1;

      ++new_added_feature_num;
    }
  }

  //printf("\033[0;33m detected: %d; matched: %d; new added feature: %d\033[0m\n",
  //    detected_new_features, matched_new_features, new_added_feature_num);

  return;
}

void ImageProcessor::pruneGridFeatures() {
  for (auto& item : *curr_features_ptr) {
    auto& grid_features = item.second;
    // Continue if the number of features in this grid does
    // not exceed the upper bound.
    if (grid_features.size() <=
        processor_config.grid_max_feature_num) continue;
    std::sort(grid_features.begin(), grid_features.end(),
        &ImageProcessor::featureCompareByLifetime);
    grid_features.erase(grid_features.begin()+
        processor_config.grid_max_feature_num,
        grid_features.end());
  }
  return;
}

void ImageProcessor::undistortPoints(
    const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics) {

  if (pts_in.size() == 0) return;

  const cv::Matx33d K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);

  const cv::Matx33d K_new(
      new_intrinsics[0], 0.0, new_intrinsics[2],
      0.0, new_intrinsics[1], new_intrinsics[3],
      0.0, 0.0, 1.0);

  if (distortion_model == "radtan") {
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
  } else if (distortion_model == "equidistant") {
    cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                                 rectification_matrix, K_new);
  } else {
    ROS_WARN_ONCE("The model %s is unrecognized, use radtan instead...",
                  distortion_model.c_str());
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
  }

  return;
}

vector<cv::Point2f> ImageProcessor::distortPoints(
    const vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const string& distortion_model,
    const cv::Vec4d& distortion_coeffs) {

  const cv::Matx33d K(intrinsics[0], 0.0, intrinsics[2],
                      0.0, intrinsics[1], intrinsics[3],
                      0.0, 0.0, 1.0);

  vector<cv::Point2f> pts_out;
  if (distortion_model == "radtan") {
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);
  } else if (distortion_model == "equidistant") {
    cv::fisheye::distortPoints(pts_in, pts_out, K, distortion_coeffs);
  } else {
    ROS_WARN_ONCE("The model %s is unrecognized, using radtan instead...",
                  distortion_model.c_str());
    vector<cv::Point3f> homogenous_pts;
    cv::convertPointsToHomogeneous(pts_in, homogenous_pts);
    cv::projectPoints(homogenous_pts, cv::Vec3d::zeros(), cv::Vec3d::zeros(), K,
                      distortion_coeffs, pts_out);
  }

  return pts_out;
}

void ImageProcessor::integrateImuData(
    Matx33f& cam0_R_p_c, Matx33f& cam1_R_p_c) {
  // Find the start and the end limit within the imu msg buffer.
  auto begin_iter = imu_msg_buffer.begin();
  while (begin_iter != imu_msg_buffer.end()) {
    if ((begin_iter->header.stamp-
          cam0_prev_img_ptr->header.stamp).toSec() < -0.01)
      ++begin_iter;
    else
      break;
  }

  auto end_iter = begin_iter;
  while (end_iter != imu_msg_buffer.end()) {
    if ((end_iter->header.stamp-
          cam0_curr_img_ptr->header.stamp).toSec() < 0.005)
      ++end_iter;
    else
      break;
  }

  // Compute the mean angular velocity in the IMU frame.
  Vec3f mean_ang_vel(0.0, 0.0, 0.0);
  for (auto iter = begin_iter; iter < end_iter; ++iter)
    mean_ang_vel += Vec3f(iter->angular_velocity.x,
        iter->angular_velocity.y, iter->angular_velocity.z);

  if (end_iter-begin_iter > 0)
    mean_ang_vel *= 1.0f / (end_iter-begin_iter);

  // Transform the mean angular velocity from the IMU
  // frame to the cam0 and cam1 frames.
  Vec3f cam0_mean_ang_vel = R_cam0_imu.t() * mean_ang_vel;
  Vec3f cam1_mean_ang_vel = R_cam1_imu.t() * mean_ang_vel;

  // Compute the relative rotation.
  double dtime = (cam0_curr_img_ptr->header.stamp-
      cam0_prev_img_ptr->header.stamp).toSec();
  Rodrigues(cam0_mean_ang_vel*dtime, cam0_R_p_c);
  Rodrigues(cam1_mean_ang_vel*dtime, cam1_R_p_c);
  cam0_R_p_c = cam0_R_p_c.t();
  cam1_R_p_c = cam1_R_p_c.t();

  // Delete the useless and used imu messages.
  imu_msg_buffer.erase(imu_msg_buffer.begin(), end_iter);
  return;
}

void ImageProcessor::rescalePoints(
    vector<Point2f>& pts1, vector<Point2f>& pts2,
    float& scaling_factor) {

  scaling_factor = 0.0f;

  for (int i = 0; i < pts1.size(); ++i) {
    scaling_factor += sqrt(pts1[i].dot(pts1[i]));
    scaling_factor += sqrt(pts2[i].dot(pts2[i]));
  }

  scaling_factor = (pts1.size()+pts2.size()) /
    scaling_factor * sqrt(2.0f);

  for (int i = 0; i < pts1.size(); ++i) {
    pts1[i] *= scaling_factor;
    pts2[i] *= scaling_factor;
  }

  return;
}

void ImageProcessor::twoPointRansac(
    const vector<Point2f>& pts1, const vector<Point2f>& pts2,
    const cv::Matx33f& R_p_c, const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    const double& inlier_error,
    const double& success_probability,
    vector<int>& inlier_markers) {

  // Check the size of input point size.
  if (pts1.size() != pts2.size())
    ROS_ERROR("Sets of different size (%lu and %lu) are used...",
        pts1.size(), pts2.size());

  double norm_pixel_unit = 2.0 / (intrinsics[0]+intrinsics[1]);
  int iter_num = static_cast<int>(
      ceil(log(1-success_probability) / log(1-0.7*0.7)));

  // Initially, mark all points as inliers.
  inlier_markers.clear();
  inlier_markers.resize(pts1.size(), 1);

  // Undistort all the points.
  vector<Point2f> pts1_undistorted(pts1.size());
  vector<Point2f> pts2_undistorted(pts2.size());
  undistortPoints(
      pts1, intrinsics, distortion_model,
      distortion_coeffs, pts1_undistorted);
  undistortPoints(
      pts2, intrinsics, distortion_model,
      distortion_coeffs, pts2_undistorted);

  // Compenstate the points in the previous image with
  // the relative rotation.
  for (auto& pt : pts1_undistorted) {
    Vec3f pt_h(pt.x, pt.y, 1.0f);
    //Vec3f pt_hc = dR * pt_h;
    Vec3f pt_hc = R_p_c * pt_h;
    pt.x = pt_hc[0];
    pt.y = pt_hc[1];
  }

  // Normalize the points to gain numerical stability.
  float scaling_factor = 0.0f;
  rescalePoints(pts1_undistorted, pts2_undistorted, scaling_factor);
  norm_pixel_unit *= scaling_factor;

  // Compute the difference between previous and current points,
  // which will be used frequently later.
  vector<Point2d> pts_diff(pts1_undistorted.size());
  for (int i = 0; i < pts1_undistorted.size(); ++i)
    pts_diff[i] = pts1_undistorted[i] - pts2_undistorted[i];

  // Mark the point pairs with large difference directly.
  // BTW, the mean distance of the rest of the point pairs
  // are computed.
  double mean_pt_distance = 0.0;
  int raw_inlier_cntr = 0;
  for (int i = 0; i < pts_diff.size(); ++i) {
    double distance = sqrt(pts_diff[i].dot(pts_diff[i]));
    // 25 pixel distance is a pretty large tolerance for normal motion.
    // However, to be used with aggressive motion, this tolerance should
    // be increased significantly to match the usage.
    if (distance > 50.0*norm_pixel_unit) {
      inlier_markers[i] = 0;
    } else {
      mean_pt_distance += distance;
      ++raw_inlier_cntr;
    }
  }
  mean_pt_distance /= raw_inlier_cntr;

  // If the current number of inliers is less than 3, just mark
  // all input as outliers. This case can happen with fast
  // rotation where very few features are tracked.
  if (raw_inlier_cntr < 3) {
    for (auto& marker : inlier_markers) marker = 0;
    return;
  }

  // Before doing 2-point RANSAC, we have to check if the motion
  // is degenerated, meaning that there is no translation between
  // the frames, in which case, the model of the RANSAC does not
  // work. If so, the distance between the matched points will
  // be almost 0.
  //if (mean_pt_distance < inlier_error*norm_pixel_unit) {
  if (mean_pt_distance < norm_pixel_unit) {
    //ROS_WARN_THROTTLE(1.0, "Degenerated motion...");
    for (int i = 0; i < pts_diff.size(); ++i) {
      if (inlier_markers[i] == 0) continue;
      if (sqrt(pts_diff[i].dot(pts_diff[i])) >
          inlier_error*norm_pixel_unit)
        inlier_markers[i] = 0;
    }
    return;
  }

  // In the case of general motion, the RANSAC model can be applied.
  // The three column corresponds to tx, ty, and tz respectively.
  MatrixXd coeff_t(pts_diff.size(), 3);
  for (int i = 0; i < pts_diff.size(); ++i) {
    coeff_t(i, 0) = pts_diff[i].y;
    coeff_t(i, 1) = -pts_diff[i].x;
    coeff_t(i, 2) = pts1_undistorted[i].x*pts2_undistorted[i].y -
      pts1_undistorted[i].y*pts2_undistorted[i].x;
  }

  vector<int> raw_inlier_idx;
  for (int i = 0; i < inlier_markers.size(); ++i) {
    if (inlier_markers[i] != 0)
      raw_inlier_idx.push_back(i);
  }

  vector<int> best_inlier_set;
  double best_error = 1e10;
  random_numbers::RandomNumberGenerator random_gen;

  for (int iter_idx = 0; iter_idx < iter_num; ++iter_idx) {
    // Randomly select two point pairs.
    // Although this is a weird way of selecting two pairs, but it
    // is able to efficiently avoid selecting repetitive pairs.
    int select_idx1 = random_gen.uniformInteger(
        0, raw_inlier_idx.size()-1);
    int select_idx_diff = random_gen.uniformInteger(
        1, raw_inlier_idx.size()-1);
    int select_idx2 = select_idx1+select_idx_diff<raw_inlier_idx.size() ?
      select_idx1+select_idx_diff :
      select_idx1+select_idx_diff-raw_inlier_idx.size();

    int pair_idx1 = raw_inlier_idx[select_idx1];
    int pair_idx2 = raw_inlier_idx[select_idx2];

    // Construct the model;
    Vector2d coeff_tx(coeff_t(pair_idx1, 0), coeff_t(pair_idx2, 0));
    Vector2d coeff_ty(coeff_t(pair_idx1, 1), coeff_t(pair_idx2, 1));
    Vector2d coeff_tz(coeff_t(pair_idx1, 2), coeff_t(pair_idx2, 2));
    vector<double> coeff_l1_norm(3);
    coeff_l1_norm[0] = coeff_tx.lpNorm<1>();
    coeff_l1_norm[1] = coeff_ty.lpNorm<1>();
    coeff_l1_norm[2] = coeff_tz.lpNorm<1>();
    int base_indicator = min_element(coeff_l1_norm.begin(),
        coeff_l1_norm.end())-coeff_l1_norm.begin();

    Vector3d model(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      Matrix2d A;
      A << coeff_ty, coeff_tz;
      Vector2d solution = A.inverse() * (-coeff_tx);
      model(0) = 1.0;
      model(1) = solution(0);
      model(2) = solution(1);
    } else if (base_indicator ==1) {
      Matrix2d A;
      A << coeff_tx, coeff_tz;
      Vector2d solution = A.inverse() * (-coeff_ty);
      model(0) = solution(0);
      model(1) = 1.0;
      model(2) = solution(1);
    } else {
      Matrix2d A;
      A << coeff_tx, coeff_ty;
      Vector2d solution = A.inverse() * (-coeff_tz);
      model(0) = solution(0);
      model(1) = solution(1);
      model(2) = 1.0;
    }

    // Find all the inliers among point pairs.
    VectorXd error = coeff_t * model;

    vector<int> inlier_set;
    for (int i = 0; i < error.rows(); ++i) {
      if (inlier_markers[i] == 0) continue;
      if (std::abs(error(i)) < inlier_error*norm_pixel_unit)
        inlier_set.push_back(i);
    }

    // If the number of inliers is small, the current
    // model is probably wrong.
    if (inlier_set.size() < 0.2*pts1_undistorted.size())
      continue;

    // Refit the model using all of the possible inliers.
    VectorXd coeff_tx_better(inlier_set.size());
    VectorXd coeff_ty_better(inlier_set.size());
    VectorXd coeff_tz_better(inlier_set.size());
    for (int i = 0; i < inlier_set.size(); ++i) {
      coeff_tx_better(i) = coeff_t(inlier_set[i], 0);
      coeff_ty_better(i) = coeff_t(inlier_set[i], 1);
      coeff_tz_better(i) = coeff_t(inlier_set[i], 2);
    }

    Vector3d model_better(0.0, 0.0, 0.0);
    if (base_indicator == 0) {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_ty_better, coeff_tz_better;
      Vector2d solution =
          (A.transpose() * A).inverse() * A.transpose() * (-coeff_tx_better);
      model_better(0) = 1.0;
      model_better(1) = solution(0);
      model_better(2) = solution(1);
    } else if (base_indicator ==1) {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_tx_better, coeff_tz_better;
      Vector2d solution =
          (A.transpose() * A).inverse() * A.transpose() * (-coeff_ty_better);
      model_better(0) = solution(0);
      model_better(1) = 1.0;
      model_better(2) = solution(1);
    } else {
      MatrixXd A(inlier_set.size(), 2);
      A << coeff_tx_better, coeff_ty_better;
      Vector2d solution =
          (A.transpose() * A).inverse() * A.transpose() * (-coeff_tz_better);
      model_better(0) = solution(0);
      model_better(1) = solution(1);
      model_better(2) = 1.0;
    }

    // Compute the error and upate the best model if possible.
    VectorXd new_error = coeff_t * model_better;

    double this_error = 0.0;
    for (const auto& inlier_idx : inlier_set)
      this_error += std::abs(new_error(inlier_idx));
    this_error /= inlier_set.size();

    if (inlier_set.size() > best_inlier_set.size()) {
      best_error = this_error;
      best_inlier_set = inlier_set;
    }
  }

  // Fill in the markers.
  inlier_markers.clear();
  inlier_markers.resize(pts1.size(), 0);
  for (const auto& inlier_idx : best_inlier_set)
    inlier_markers[inlier_idx] = 1;

  //printf("inlier ratio: %lu/%lu\n",
  //    best_inlier_set.size(), inlier_markers.size());

  return;
}

void ImageProcessor::publish() {

  // Publish features.
//  CameraMeasurementPtr feature_msg_ptr(new CameraMeasurement);
//  feature_msg_ptr->header.stamp = cam0_curr_img_ptr->header.stamp;

  vector<FeatureIDType> curr_ids(0);
  vector<Point2f> curr_cam0_points(0);
  vector<Point2f> curr_cam1_points(0);

  for (const auto& grid_features : (*curr_features_ptr)) {
    for (const auto& feature : grid_features.second) {
      curr_ids.push_back(feature.id);
      curr_cam0_points.push_back(feature.cam0_point);
      curr_cam1_points.push_back(feature.cam1_point);
    }
  }

  vector<Point2f> curr_cam0_points_undistorted(0);
  vector<Point2f> curr_cam1_points_undistorted(0);

  undistortPoints(
      curr_cam0_points, cam0_intrinsics, cam0_distortion_model,
      cam0_distortion_coeffs, curr_cam0_points_undistorted);
  undistortPoints(
      curr_cam1_points, cam1_intrinsics, cam1_distortion_model,
      cam1_distortion_coeffs, curr_cam1_points_undistorted);
#if 0
  for (int i = 0; i < curr_ids.size(); ++i) {
      int id = curr_ids[i];
      Eigen::Vector2d a(curr_cam0_points[i].x, curr_cam0_points[i].y);
      Eigen::Vector3d b;
      m_camera[0]->liftProjective(a, b);

      geometry_msgs::Point32 p;
      p.x = b.x() / b.z();
      p.y = b.y() / b.z();
      p.z = 1;

      feature_points->points.push_back(p);
      id_of_point.values.push_back(id);
      u_of_point.values.push_back(curr_cam0_points[i].x);
      v_of_point.values.push_back(curr_cam0_points[i].y);

      // right camera
      Eigen::Vector2d a1(curr_cam1_points[i].x, curr_cam1_points[i].y);
      Eigen::Vector3d b1;
      m_camera[1]->liftProjective(a1, b1);
      
      geometry_msgs::Point32 p1;
      p1.x = b1.x() / b1.z();
      p1.y = b1.y() / b1.z();
      p1.z = 1;

      feature_points->points.push_back(p1);
      id_of_point.values.push_back(id);
      u_of_point.values.push_back(curr_cam1_points[i].x);
      v_of_point.values.push_back(curr_cam1_points[i].y);
  }
  feature_points->channels.push_back(id_of_point);
  feature_points->channels.push_back(u_of_point);
  feature_points->channels.push_back(v_of_point);
  pub_img.publish(feature_points);
#endif
sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;
	sensor_msgs::ChannelFloat32 point_3D;

        feature_points->header = cam0_curr_img_ptr->header;
        feature_points->header.frame_id = "world";
  for (int i = 0; i < curr_ids.size(); ++i) {
    float disparity = curr_cam0_points[i].x - curr_cam1_points[i].x;
    if(disparity<0){
	disparity = -1*disparity;
      }
    if(disparity >= 0 && disparity < cam0_intrinsics[0])
    {
      float Depth;
#if 0
      if(disparity<0){
	disparity = -1*disparity;
      }
#endif
      if(disparity == 0)
      {
	  disparity=0.01;
//	  bestuR = uL-0.01;
	  continue;
      }
      Depth = mbf/disparity;
      geometry_msgs::Point32 p;
      p.x = curr_cam0_points_undistorted[i].x;
      p.y = curr_cam0_points_undistorted[i].y;
      p.z = Depth;
      feature_points->points.push_back(p);
      id_of_point.values.push_back(curr_ids[i]);
      id_of_point.values.push_back(curr_ids[i]);
      geometry_msgs::Point32 p1;
      p1.x = curr_cam1_points_undistorted[i].x;
      p1.y = curr_cam1_points_undistorted[i].y;
      p1.z = 1;//Depth;
      feature_points->points.push_back(p1);
      u_of_point.values.push_back(curr_cam0_points[i].x);
      v_of_point.values.push_back(curr_cam0_points[i].y);
      u_of_point.values.push_back(curr_cam1_points[i].x);
      v_of_point.values.push_back(curr_cam1_points[i].y);
      point_3D.values.push_back(Depth/cam0_intrinsics[0]*(curr_cam0_points[i].x - cam0_intrinsics[2]));
      point_3D.values.push_back(Depth/cam0_intrinsics[0]*(curr_cam0_points[i].y - cam0_intrinsics[3]));
    }
  }
#if 0
  for (int i = 0; i < curr_ids.size(); ++i) {
    geometry_msgs::Point32 p;
    p.x = curr_cam0_points_undistorted[i].x;
    p.y = curr_cam0_points_undistorted[i].y;
//    float disparity = (curr_cam0_points_undistorted[i].x - curr_cam1_points_undistorted[i].x);
//    p.z = mbf/disparity;
    p.z = 1;
    feature_points->points.push_back(p);
    geometry_msgs::Point32 p1;
    p1.x = curr_cam1_points_undistorted[i].x;
    p1.y = curr_cam1_points_undistorted[i].y;
    p1.z = 1;
    feature_points->points.push_back(p1);
    id_of_point.values.push_back(curr_ids[i]);
    id_of_point.values.push_back(curr_ids[i]);
    u_of_point.values.push_back(curr_cam0_points[i].x);
    v_of_point.values.push_back(curr_cam0_points[i].y);
    u_of_point.values.push_back(curr_cam1_points[i].x);
    v_of_point.values.push_back(curr_cam1_points[i].y);
  }
#endif
    feature_points->channels.push_back(id_of_point);
    feature_points->channels.push_back(u_of_point);
    feature_points->channels.push_back(v_of_point);
    feature_points->channels.push_back(point_3D);
  pub_img.publish(feature_points);
/*
  // Publish tracking info.
  TrackingInfoPtr tracking_info_msg_ptr(new TrackingInfo());
  tracking_info_msg_ptr->header.stamp = cam0_curr_img_ptr->header.stamp;
  tracking_info_msg_ptr->before_tracking = before_tracking;
  tracking_info_msg_ptr->after_tracking = after_tracking;
  tracking_info_msg_ptr->after_matching = after_matching;
  tracking_info_msg_ptr->after_ransac = after_ransac;
  tracking_info_pub.publish(tracking_info_msg_ptr);
*/
  return;
}

void ImageProcessor::drawFeaturesMono() {
  // Colors for different features.
  Scalar tracked(0, 255, 0);
  Scalar new_feature(0, 255, 255);

  static int grid_height =
    cam0_curr_img_ptr->image.rows / processor_config.grid_row;
  static int grid_width =
    cam0_curr_img_ptr->image.cols / processor_config.grid_col;

  // Create an output image.
  int img_height = cam0_curr_img_ptr->image.rows;
  int img_width = cam0_curr_img_ptr->image.cols;
  Mat out_img(img_height, img_width, CV_8UC3);
  cvtColor(cam0_curr_img_ptr->image, out_img, CV_GRAY2RGB);

  // Draw grids on the image.
  for (int i = 1; i < processor_config.grid_row; ++i) {
    Point pt1(0, i*grid_height);
    Point pt2(img_width, i*grid_height);
    line(out_img, pt1, pt2, Scalar(255, 0, 0));
  }
  for (int i = 1; i < processor_config.grid_col; ++i) {
    Point pt1(i*grid_width, 0);
    Point pt2(i*grid_width, img_height);
    line(out_img, pt1, pt2, Scalar(255, 0, 0));
  }

  // Collect features ids in the previous frame.
  vector<FeatureIDType> prev_ids(0);
  for (const auto& grid_features : *prev_features_ptr)
    for (const auto& feature : grid_features.second)
      prev_ids.push_back(feature.id);

  // Collect feature points in the previous frame.
  map<FeatureIDType, Point2f> prev_points;
  for (const auto& grid_features : *prev_features_ptr)
    for (const auto& feature : grid_features.second)
      prev_points[feature.id] = feature.cam0_point;

  // Collect feature points in the current frame.
  map<FeatureIDType, Point2f> curr_points;
  for (const auto& grid_features : *curr_features_ptr)
    for (const auto& feature : grid_features.second)
      curr_points[feature.id] = feature.cam0_point;

  // Draw tracked features.
  for (const auto& id : prev_ids) {
    if (prev_points.find(id) != prev_points.end() &&
        curr_points.find(id) != curr_points.end()) {
      cv::Point2f prev_pt = prev_points[id];
      cv::Point2f curr_pt = curr_points[id];
      circle(out_img, curr_pt, 3, tracked);
      line(out_img, prev_pt, curr_pt, tracked, 1);

      prev_points.erase(id);
      curr_points.erase(id);
    }
  }

  // Draw new features.
  for (const auto& new_curr_point : curr_points) {
    cv::Point2f pt = new_curr_point.second;
    circle(out_img, pt, 3, new_feature, -1);
  }

  imshow("Feature", out_img);
  waitKey(5);
}

void ImageProcessor::drawFeaturesStereo() {

  if(pub_match.getNumSubscribers() > 0)
  {
    // Colors for different features.
    Scalar tracked(0, 255, 0);
    Scalar new_feature(0, 255, 255);

    static int grid_height =
      cam0_curr_img_ptr->image.rows / processor_config.grid_row;
    static int grid_width =
      cam0_curr_img_ptr->image.cols / processor_config.grid_col;

    // Create an output image.
    int img_height = cam0_curr_img_ptr->image.rows;
    int img_width = cam0_curr_img_ptr->image.cols;
    Mat out_img(img_height, img_width*2, CV_8UC3);
    cvtColor(cam0_curr_img_ptr->image,
             out_img.colRange(0, img_width), CV_GRAY2RGB);
    cvtColor(cam1_curr_img_ptr->image,
             out_img.colRange(img_width, img_width*2), CV_GRAY2RGB);

    // Draw grids on the image.
    for (int i = 1; i < processor_config.grid_row; ++i) {
      Point pt1(0, i*grid_height);
      Point pt2(img_width*2, i*grid_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i) {
      Point pt1(i*grid_width, 0);
      Point pt2(i*grid_width, img_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }
    for (int i = 1; i < processor_config.grid_col; ++i) {
      Point pt1(i*grid_width+img_width, 0);
      Point pt2(i*grid_width+img_width, img_height);
      line(out_img, pt1, pt2, Scalar(255, 0, 0));
    }

    // Collect features ids in the previous frame.
    vector<FeatureIDType> prev_ids(0);
    for (const auto& grid_features : *prev_features_ptr)
      for (const auto& feature : grid_features.second)
        prev_ids.push_back(feature.id);

    // Collect feature points in the previous frame.
    map<FeatureIDType, Point2f> prev_cam0_points;
    map<FeatureIDType, Point2f> prev_cam1_points;
    for (const auto& grid_features : *prev_features_ptr)
      for (const auto& feature : grid_features.second) {
        prev_cam0_points[feature.id] = feature.cam0_point;
        prev_cam1_points[feature.id] = feature.cam1_point;
      }

    // Collect feature points in the current frame.
    map<FeatureIDType, Point2f> curr_cam0_points;
    map<FeatureIDType, Point2f> curr_cam1_points;
    for (const auto& grid_features : *curr_features_ptr)
      for (const auto& feature : grid_features.second) {
        curr_cam0_points[feature.id] = feature.cam0_point;
        curr_cam1_points[feature.id] = feature.cam1_point;
      }

    // Draw tracked features.
    for (const auto& id : prev_ids) {
      if (prev_cam0_points.find(id) != prev_cam0_points.end() &&
          curr_cam0_points.find(id) != curr_cam0_points.end()) {
        cv::Point2f prev_pt0 = prev_cam0_points[id];
        cv::Point2f prev_pt1 = prev_cam1_points[id] + Point2f(img_width, 0.0);
        cv::Point2f curr_pt0 = curr_cam0_points[id];
        cv::Point2f curr_pt1 = curr_cam1_points[id] + Point2f(img_width, 0.0);

        circle(out_img, curr_pt0, 3, tracked, -1);
        circle(out_img, curr_pt1, 3, tracked, -1);
        line(out_img, prev_pt0, curr_pt0, tracked, 1);
        line(out_img, prev_pt1, curr_pt1, tracked, 1);

        prev_cam0_points.erase(id);
        prev_cam1_points.erase(id);
        curr_cam0_points.erase(id);
        curr_cam1_points.erase(id);
      }
    }

    // Draw new features.
    for (const auto& new_cam0_point : curr_cam0_points) {
      cv::Point2f pt0 = new_cam0_point.second;
      cv::Point2f pt1 = curr_cam1_points[new_cam0_point.first] +
        Point2f(img_width, 0.0);

      circle(out_img, pt0, 3, new_feature, -1);
      circle(out_img, pt1, 3, new_feature, -1);
    }

    cv_bridge::CvImage debug_image(cam0_curr_img_ptr->header, "bgr8", out_img);
    pub_match.publish(debug_image.toImageMsg());
  }
  //imshow("Feature", out_img);
  //waitKey(5);

  return;
}

void ImageProcessor::updateFeatureLifetime() {
  for (int code = 0; code <
      processor_config.grid_row*processor_config.grid_col; ++code) {
    vector<FeatureMetaData>& features = (*curr_features_ptr)[code];
    for (const auto& feature : features) {
      if (feature_lifetime.find(feature.id) == feature_lifetime.end())
        feature_lifetime[feature.id] = 1;
      else
        ++feature_lifetime[feature.id];
    }
  }

  return;
}

void ImageProcessor::featureLifetimeStatistics() {

  map<int, int> lifetime_statistics;
  for (const auto& data : feature_lifetime) {
    if (lifetime_statistics.find(data.second) ==
        lifetime_statistics.end())
      lifetime_statistics[data.second] = 1;
    else
      ++lifetime_statistics[data.second];
  }

  for (const auto& data : lifetime_statistics)
    cout << data.first << " : " << data.second << endl;

  return;
}

#if 0
void img_callback(const sensor_msgs::ImageConstPtr& cam0_img,
    const sensor_msgs::ImageConstPtr& cam1_img)
{
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = cam0_img->header.stamp.toSec();
        last_image_time = cam0_img->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (cam0_img->header.stamp.toSec() - last_image_time > 1.0 || cam0_img->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = cam0_img->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (cam0_img->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (cam0_img->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = cam0_img->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;
    if (cam0_img->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = cam0_img->header;
        img.height = cam0_img->height;
        img.width = cam0_img->width;
        img.is_bigendian = cam0_img->is_bigendian;
        img.step = cam0_img->step;
        img.data = cam0_img->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(cam0_img, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cam0_img->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = cam0_img->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}
#endif
int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    ImageProcessor igb;
    igb.loadParameters(n);
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }
    
    message_filters::Subscriber<sensor_msgs::Image> cam0_img_sub(n, "/cam0/image_raw", 2000);
    message_filters::Subscriber<sensor_msgs::Image> cam1_img_sub(n, "/cam1/image_raw", 2000);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(2000), cam0_img_sub,cam1_img_sub);
    sync.registerCallback(boost::bind(&ImageProcessor::stereoCallback,&igb,_1,_2));
/*    
    message_filters::Subscriber<
    sensor_msgs::Image> cam0_img_sub(n, "cam0_image", 10);
  message_filters::Subscriber<
    sensor_msgs::Image> cam1_img_sub(n, "cam1_image", 10);
  message_filters::TimeSynchronizer<
    sensor_msgs::Image, sensor_msgs::Image> stereo_sub(cam0_img_sub, cam1_img_sub, 10);
  stereo_sub.registerCallback(boost::bind(&img_callback, _1, _2));*/
//    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?