# Published on /world_model by the world model node.
std_msgs/Header header              # map frame
TrashItem[] items
nav_msgs/OccupancyGrid coverage     # 100 = swept by the camera footprint, 0 = not yet, -1 = outside the geofence
string[] compartment_recyclable     # item ids, in storing order
string[] compartment_residual
float32 basket_load_kg
