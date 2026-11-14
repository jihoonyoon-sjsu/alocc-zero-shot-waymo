std_msgs/Header header

# True when a mock generator produced this message. This is not ALOcc inference.
bool is_mock

# Connector name: mock, file, python, ros, or pkl.
string prediction_source

# False when the source had no sensor timestamp. header.stamp is then 0 and is not a capture time.
bool timestamp_available

# False when resolution and origin are not physical quantities.
bool geometry_confirmed

# Position in the source sequence. This is not a timestamp.
uint32 sample_index

# Short source fields. Empty for mock data.
string sample_metadata

# Wall-clock seconds when framework handling of this sample started.
# This is not a dataset timestamp.
float64 pipeline_publish_time_s

uint32 dim_x
uint32 dim_y
uint32 dim_z

float32 resolution_x
float32 resolution_y
float32 resolution_z

float64 origin_x
float64 origin_y
float64 origin_z

string coordinate_convention

# Length must equal dim_x * dim_y * dim_z * 3.
# Each voxel contributes vx, vy, vz. x changes fastest, then y, then z.
float32[] flow_vectors

# Units are configuration. They are not a confirmed ALOcc flow definition.
string flow_units

string vector_layout
