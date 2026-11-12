std_msgs/Header header

# True when a mock generator produced this message. This is not ALOcc inference.
bool is_mock

# Connector name: mock, file, python, ros, or pkl.
string prediction_source

# False when the source had no sensor timestamp. header.stamp is then 0 and is not a capture time.
bool timestamp_available

# False when resolution and origin are not physical quantities.
bool geometry_confirmed

# False when this occupancy sample has no companion flow prediction.
bool flow_supplied

# Position in the source sequence. This is not a timestamp.
uint32 sample_index

# Short source fields such as dataset, scene, and frame. Empty for mock data.
string sample_metadata

# Wall-clock seconds when framework handling of this sample started.
# This is not a dataset timestamp. It is only for pipeline latency.
float64 pipeline_publish_time_s

# Voxel counts. semantic_labels length must equal dim_x * dim_y * dim_z.
uint32 dim_x
uint32 dim_y
uint32 dim_z

# Voxel size in meters. Confirmed ALOcc pickle playback uses 0.4 m on each axis.
float32 resolution_x
float32 resolution_y
float32 resolution_z

# Minimum corner of voxel (0, 0, 0), in meters, when geometry_confirmed is true.
float64 origin_x
float64 origin_y
float64 origin_z

# Confirmed occupancy uses ego_x_forward_y_left_z_up. The ROS frame id is ego.
string coordinate_convention

# semantic_labels stores an index into this list.
string[] class_names

# Row-major class index. x changes fastest, then y, then z.
uint8[] semantic_labels

string label_layout
