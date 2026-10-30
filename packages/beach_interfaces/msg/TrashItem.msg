# One object in the world model (notes/architecture.md, /world_model).
uint8 SEEN = 0        # detected, not collected yet
uint8 COLLECTED = 1   # in a basket compartment
uint8 PLACED = 2      # released into a bin
uint8 SKIPPED = 3     # given up (see skip_reason)

string id                           # world-model id, e.g. wm_007
string scene_id                     # id from bench/scenes.yaml if known (ground truth only), else empty
string object_class                 # trash | distractor
string material                     # pet | metal | glass | residual; empty until classified
string bin_id                       # recyclable | residual: compartment (COLLECTED) or bin (PLACED); else empty
geometry_msgs/PoseStamped pose      # map frame
float32 confidence                  # detection confidence
uint8 status
string skip_reason                  # e.g. unreachable, grasp_failed, no_progress, distractor
uint32 times_seen
builtin_interfaces/Time last_seen
