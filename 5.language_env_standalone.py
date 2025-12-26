#!/usr/bin/env python3
"""
Standalone script to collect multi-task demonstration data with keyboard teleop
Uses SimpleEnv2 for language/task instructions
Run from terminal: python 5.language_env_standalone.py
"""
import sys
import random
import numpy as np
import os
os.environ["MUJOCO_GL"] = "glfw"
from PIL import Image
from mujoco_env.y_env2 import SimpleEnv2
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import shutil

# Configuration
TASK_NAME = 'Pick and place with language instructions'
SEED = 0
REPO_NAME = 'omy_pnp_language'
NUM_DEMO = 20  # Number of demonstrations to collect
ROOT = "./demo_data_language"
xml_path = 'asset/example_scene_y2.xml'

print("=" * 60)
print("Starting Multi-Task Data Collection with Language Instructions")
print("=" * 60)

# Create dataset
create_new = True
if os.path.exists(ROOT):
    print(f"Directory {ROOT} already exists.")
    ans = input("Do you want to delete it? (y/n) ")
    if ans.lower() == 'y':
        shutil.rmtree(ROOT)
    else:
        create_new = False

if create_new:
    print("Creating new dataset...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=ROOT, 
        robot_type="omy",
        fps=20,
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["obj_init"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
else:
    print("Loading from previous dataset...")
    dataset = LeRobotDataset(REPO_NAME, root=ROOT)

# Initialize environment with viewer
print("\nInitializing environment with viewer...")
PnPEnv = SimpleEnv2(xml_path, seed=SEED, state_type='joint_angle', init_viewer=True)

print("\n" + "=" * 60)
print("KEYBOARD CONTROLS:")
print("=" * 60)
print("XY Plane: W(backward) S(forward) A(left) D(right)")
print("Z Axis:   R(up) F(down)")
print("Rotation: Q(tilt left) E(tilt right)")
print("          UP/DOWN/LEFT/RIGHT arrows for other rotations")
print("SPACEBAR: Toggle gripper")
print("Z key:    Reset environment")
print("=" * 60)
print(f"\nCollecting {NUM_DEMO} demonstration(s) with language instructions...")
print("Remember: Release gripper and move up to get success signal!")
print("=" * 60 + "\n")

# Main collection loop
action = np.zeros(7)
episode_id = 0
record_flag = False

try:
    while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
        PnPEnv.step_env()
        
        if PnPEnv.env.loop_every(HZ=20):
            # Check if episode is done
            done = PnPEnv.check_success()
            if done:
                print(f"\n✓ Episode {episode_id + 1} completed successfully!")
                dataset.save_episode()
                dataset.clear_episode_buffer()
                PnPEnv.reset()
                episode_id += 1
                record_flag = False
                
            # Get keyboard input
            action, reset = PnPEnv.teleop_robot()
            
            # Start recording when robot moves
            # Use np.any() instead of sum() to avoid sign cancellation
            if not record_flag and np.any(action != 0):
                record_flag = True
                print(f"▶ Recording started... Task: {PnPEnv.instruction}")
            
            # Handle reset
            if reset:
                print("↺ Reset environment")
                PnPEnv.reset()
                dataset.clear_episode_buffer()
                record_flag = False
            
            # Get images from environment
            agent_image, wrist_image = PnPEnv.grab_image()
            
            # Resize images
            agent_image = Image.fromarray(agent_image)
            wrist_image = Image.fromarray(wrist_image)
            agent_image = agent_image.resize((256, 256))
            wrist_image = wrist_image.resize((256, 256))
            agent_image = np.array(agent_image)
            wrist_image = np.array(wrist_image)
            
            # Step environment
            joint_q = PnPEnv.step(action)
            
            # Get action (6 joint angles + 1 gripper)
            action_to_save = PnPEnv.q[:7].astype(np.float32)
            
            # Add frame to dataset
            if record_flag:
                dataset.add_frame({
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": joint_q[:6], 
                    "action": action_to_save,
                    "obj_init": PnPEnv.obj_init_pose,
                }, task=PnPEnv.instruction)
            
            # Render
            PnPEnv.render(teleop=True, idx=episode_id)

except KeyboardInterrupt:
    print("\n\nCollection interrupted by user.")

# Cleanup
print("\nClosing viewer...")
PnPEnv.env.close_viewer()

# NOTE: Do NOT delete images/ folder automatically!
# LeRobotDataset may store image paths in parquet files.
# If images are stored as files (not embedded in parquet),
# deleting them will break future dataset loads.

print("\n" + "=" * 60)
print("✓ Data collection complete!")
print(f"Dataset saved to: {ROOT}")
print(f"Total episodes collected: {episode_id}")
print("\n⚠️  Note: 'images/' folder preserved for safety.")
print("   After confirming training works, you may delete it manually.")
print("=" * 60)
