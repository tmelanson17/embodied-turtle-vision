# embodied-turtle-vision


## Component breakdown

This code base consists of several subparts:

- Perception : parsing data through sensors such as YOLO

- Planning : converting perception output to a plan (usually one of a discrete set of outputs)

- Controls : converting the high level plan into ROS motor commands (/cmd\_vel)

