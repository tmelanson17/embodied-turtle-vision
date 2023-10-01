cmake_minimum_required(VERSION 3.8)
project(discrete_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_executable(signal_to_cmd_vel src/signal_to_cmd_vel.cpp)
add_executable(minimal_publisher src/minimal_publisher.cpp)
ament_target_dependencies(signal_to_cmd_vel rclcpp geometry_msgs)
ament_target_dependencies(minimal_publisher rclcpp geometry_msgs)
target_compile_features(signal_to_cmd_vel PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_compile_features(minimal_publisher PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS signal_to_cmd_vel
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS minimal_publisher
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
