/*
 * Copyright 2018 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cartographer/common/math.h"
#include "cartographer/io/proto_stream.h"
#include "cartographer/mapping/proto/pose_graph.pb.h"
#include "cartographer/mapping/proto/trajectory_builder_options.pb.h"
#include "cartographer/transform/transform_interpolation_buffer.h"
#include "cartographer_ros/msg_conversion.h"
#include "cartographer_ros/time_conversion.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "ros/ros.h"
#include "ros/time.h"
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "tf2_eigen/tf2_eigen.h"
#include "tf2_msgs/TFMessage.h"

DEFINE_double(jump, 1, "");
DEFINE_string(bag_filename, "",
              "Bags to process, must be in the same order as the trajectories "
              "in 'pose_graph_filename'.");
DEFINE_string(pbstream_filename, "",
              "Proto stream file containing the pose graph.");

namespace cartographer_ros {
namespace {

double FractionSmallerThan(const std::vector<double>& v, double x) {
  return static_cast<double>(std::count_if(
             v.begin(), v.end(), [=](double value) { return value < x; })) /
         v.size();
}

std::string QuantilesToString(std::vector<double>* v) {
  if (v->empty()) return "(empty vector)";
  std::sort(v->begin(), v->end());
  std::stringstream result;
  const int kNumQuantiles = 10;
  for (int i = 0; i < kNumQuantiles; ++i) {
    auto value = v->at(v->size() * i / kNumQuantiles);
    auto percentage = 100 * i / kNumQuantiles;
    result << percentage << "%," << value << "\n";
  }
  result << "100%: " << v->back() << "\n";
  return result.str();
}

void Run(const std::string& pbstream_filename,
         const std::string& bag_filename) {
  std::unique_ptr<std::ostream> log_info = ::cartographer::common::make_unique<
      std::ofstream>(
      "/usr/local/google/home/pifon/workspace/bags/localization/trajectory.txt",
      std::ios_base::out);

  std::map<cartographer::common::Time, std::pair<double, double>> results;
  cartographer::io::ProtoStreamReader reader(pbstream_filename);
  cartographer::mapping::proto::PoseGraph pose_graph_proto;
  CHECK(reader.ReadProto(&pose_graph_proto));
  const cartographer::mapping::proto::Trajectory& last_trajectory_proto =
      *pose_graph_proto.mutable_trajectory()->rbegin();
  const cartographer::transform::TransformInterpolationBuffer
      transform_interpolation_buffer(last_trajectory_proto);

  const cartographer::common::Time trajectory_start =
      cartographer::common::FromUniversal(
          last_trajectory_proto.node().begin()->timestamp());
  LOG(INFO) << last_trajectory_proto.node().begin()->node_index();

  const cartographer::common::Time trajectory_end =
      cartographer::common::FromUniversal(
          last_trajectory_proto.node().rbegin()->timestamp());
  LOG(INFO) << last_trajectory_proto.node().rbegin()->node_index();
  LOG(INFO) << "Allow only timestamps between " << trajectory_start << " and "
            << trajectory_end;

  rosbag::Bag bag;
  bag.open(bag_filename, rosbag::bagmode::Read);
  rosbag::View view(bag);
  std::vector<double> deviation_translation, deviation_rotation;
  bool jump_found = false;
  bool transform_initialized = false;
  double kThreshold = FLAGS_jump;
  cartographer::common::Time eval_start, eval_end;
  double max_jump = 0;
  cartographer::transform::Rigid3d prev_transform;
  const double signal_maximum = std::numeric_limits<double>::max();
  for (const rosbag::MessageInstance& message : view) {
    if (!message.isType<tf2_msgs::TFMessage>()) {
      continue;
    }
    auto tf_message = message.instantiate<tf2_msgs::TFMessage>();
    for (const auto& transform : tf_message->transforms) {
      if (transform.header.frame_id != "map" ||
          transform.child_frame_id != "base_link") {
        continue;
      }

      const cartographer::common::Time transform_time =
          FromRos(message.getTime());
      if (transform_time > trajectory_end ||
          transform_time < trajectory_start) {
        continue;
      }
      if (!transform_interpolation_buffer.Has(transform_time)) {
        deviation_translation.push_back(signal_maximum);
        deviation_rotation.push_back(signal_maximum);
        continue;
      }
      auto published_transform = ToRigid3d(transform);
      if (!jump_found) {
        if (!transform_initialized) {
          prev_transform = published_transform;
          transform_initialized = true;
        }
        double jump =
            (published_transform.translation() - prev_transform.translation())
                .norm();
        if (jump > max_jump) {
          max_jump = jump;
          LOG(INFO) << "Jump = " << jump << " at time = "
                    << cartographer::common::ToUniversal(transform_time);
        }
        if (jump < kThreshold) {
          prev_transform = published_transform;
          continue;
        }
        jump_found = true;
        eval_start = transform_time;
      }
      eval_end = transform_time;
      auto optimized_transform =
          transform_interpolation_buffer.Lookup(transform_time);
      if (cartographer::common::ToUniversal(transform_time) <
          635415433949111821) {
        continue;
      }
      deviation_translation.push_back((published_transform.translation() -
                                       optimized_transform.translation())
                                          .norm());
      deviation_rotation.push_back(
          published_transform.rotation().angularDistance(
              optimized_transform.rotation()));

      results[transform_time] = std::make_pair(deviation_translation.back(),
                                               deviation_rotation.back());
    }
  }
  bag.close();
  (*log_info) << "Distribution of translation difference:\n"
              << QuantilesToString(&deviation_translation) << std::endl;
  (*log_info) << "Distribution of rotation difference:\n"
              << QuantilesToString(&deviation_rotation) << std::endl;
  (*log_info) << "Fraction of translation difference smaller than 1m: "
              << FractionSmallerThan(deviation_translation, 1) << std::endl;
  (*log_info) << "Fraction of translation difference smaller than 0.1m: "
              << FractionSmallerThan(deviation_translation, 0.1) << std::endl;
  (*log_info) << "Fraction of translation difference smaller than 0.05m: "
              << FractionSmallerThan(deviation_translation, 0.05) << std::endl;
  (*log_info) << "Fraction of translation difference smaller than 0.01m: "
              << FractionSmallerThan(deviation_translation, 0.01) << std::endl;
  (*log_info) << "Start = " << cartographer::common::ToUniversal(eval_start)
              << std::endl;
  (*log_info) << "End = " << cartographer::common::ToUniversal(eval_end)
              << std::endl;
  (*log_info) << "Duration = "
              << cartographer::common::ToSeconds(eval_end - eval_start)
              << std::endl;
  (*log_info) << "Real Duration = "
              << cartographer::common::ToSeconds(
                     eval_end -
                     cartographer::common::FromUniversal(635415433949111821))
              << std::endl;

  LOG(INFO) << "Biggest = " << max_jump << " at time = " << eval_start;
  for (const auto& item : results) {
    (*log_info) << item.first << "," << item.second.first << ","
                << item.second.second << std::endl;
  }
}

}  // namespace
}  // namespace cartographer_ros

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::SetUsageMessage(
      "\n\n"
      "This compares a trajectory from a bag file against the "
      "last trajectory in a pbstream file.\n");
  google::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(!FLAGS_bag_filename.empty()) << "-bag_filename is missing.";
  CHECK(!FLAGS_pbstream_filename.empty()) << "-pbstream_filename is missing.";
  ::cartographer_ros::Run(FLAGS_pbstream_filename, FLAGS_bag_filename);
}
