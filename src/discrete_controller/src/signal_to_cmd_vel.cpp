#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "geometry_msgs/msg/twist.hpp"

enum Direction {
    LEFT=0,
    RIGHT=1,
    FORWARD=2,
    STOP=3
};

class DiscreteController : public rclcpp::Node
{
public:
    DiscreteController() : Node("discrete_controller")
    {
        // Create a subscriber to listen for integer messages
        subscriber_ = this->create_subscription<std_msgs::msg::Int32>(
            "/direction", 10, std::bind(&DiscreteController::integerCallback, this, std::placeholders::_1));

        // Create a publisher to publish Twist messages
        twist_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10);

        RCLCPP_INFO(this->get_logger(), "DiscreteController is ready.");
    }

private:
    void integerCallback(const std_msgs::msg::Int32::SharedPtr msg)
    {
	RCLCPP_INFO(this->get_logger(), "Message received: %d", msg->data);
        // Extract the integer value from the message
        Direction dir = static_cast<Direction>(msg->data);

        // Create a Twist message
        geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x = 0.0; 

        // Set angular velocity (no rotation)
        twist_msg.angular.z = 0.0;


	double LIN_VEL=0.5;
	double ANG_VEL=1.0;

        // Set linear velocity based on the received integer value
	switch (dir) {
	    case (Direction::LEFT): {
                twist_msg.angular.z = -ANG_VEL;
		break;
	    }
	    case (Direction::RIGHT): {
                twist_msg.angular.z = -ANG_VEL;
		break;
	    }
	    case (Direction::FORWARD): {
                twist_msg.linear.x = LIN_VEL;
		break;
	    }
	    case (Direction::STOP): {
		break;
	    }
	}

        // Publish the Twist message
        twist_publisher_->publish(twist_msg);
    }

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscriber_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DiscreteController>());
    rclcpp::shutdown();
    return 0;
}

