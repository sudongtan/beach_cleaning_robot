// A lower-rate copy of Gazebo's /clock for our own ROS nodes.
//
// Gazebo publishes a clock message every physics step (1 kHz). Every node with use_sim_time
// processes each one; with ten Python nodes that cost ~3.5 CPU threads and slowed the
// simulation to RTF ~0.25 (notes/logs/A2.md). This node forwards a message only when sim time
// has advanced by `period_s` (default 0.01 s = 100 Hz in sim time); our nodes remap /clock to
// its output. /clock itself stays at 1 kHz: the leg controller (legubiao, inside Gazebo's
// controller manager) reads the ROS node clock and failed to stand up with a 100 Hz /clock
// (A2b, 2026-10-31). C++ because a Python relay would itself pay the 1 kHz cost.
//
//   /clock (from ros_gz_bridge, every step) -> /sim/clock_throttled (every period_s of sim time)

#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rosgraph_msgs/msg/clock.hpp"

class ClockThrottle : public rclcpp::Node
{
public:
  ClockThrottle()
  : Node("clock_throttle")
  {
    period_ns_ = static_cast<int64_t>(declare_parameter("period_s", 0.01) * 1e9);
    // reliable: compatible with best-effort subscribers (rclpy, rclcpp TimeSource) and reliable ones
    pub_ = create_publisher<rosgraph_msgs::msg::Clock>("/sim/clock_throttled", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    sub_ = create_subscription<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::ClockQoS(),
      [this](rosgraph_msgs::msg::Clock::ConstSharedPtr msg) {
        const int64_t t = rclcpp::Time(msg->clock).nanoseconds();
        // forward the first message, every period, and any jump back (simulation reset)
        if (last_ns_ < 0 || t - last_ns_ >= period_ns_ || t < last_ns_) {
          last_ns_ = t;
          pub_->publish(*msg);
        }
      });
  }

private:
  int64_t period_ns_;
  int64_t last_ns_ = -1;
  rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr pub_;
  rclcpp::Subscription<rosgraph_msgs::msg::Clock>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClockThrottle>());
  rclcpp::shutdown();
  return 0;
}
