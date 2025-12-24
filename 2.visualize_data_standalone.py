#!/usr/bin/env python3
"""
Standalone script to visualize collected demonstration data
Run from terminal: python visualize_data_standalone.py
"""
import torch
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_json, serialize_dict
from mujoco_env.y_env import SimpleEnv

print("=" * 60)
print("Visualizing Collected Demonstration Data")
print("=" * 60)

# Configuration
DATASET_ROOT = './demo_data'  # Change to './demo_data_example' for example data
DATASET_REPO = 'omy_pnp'
XML_PATH = './asset/example_scene_y.xml'

# Load dataset
print(f"\nLoading dataset from {DATASET_ROOT}...")
try:
    dataset = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)
    print(f"✓ Dataset loaded successfully")
    print(f"  Total episodes: {len(dataset.episode_data_index['from'])}")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print(f"  Make sure you've collected data using collect_data_standalone.py first")
    exit(1)

# Episode sampler
class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for a single episode"""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

# Create episode sampler and dataloader
print(f"\nPreparing episodes for visualization...")
total_episodes = len(dataset.episode_data_index['from'])
print(f"✓ Total episodes available: {total_episodes}")

# Initialize environment with viewer
print(f"\nInitializing MuJoCo environment...")
PnPEnv = SimpleEnv(XML_PATH, action_type='joint_angle', init_viewer=True)
print("✓ Environment initialized")

print("\n" + "=" * 60)
print(f"Playing all {total_episodes} demonstration replays...")
print("Close the viewer window to exit")
print("=" * 60 + "\n")

# Visualization loop for all episodes
try:
    for episode_index in range(total_episodes):
        print(f"\n▶ Episode {episode_index + 1}/{total_episodes}")
        
        episode_sampler = EpisodeSampler(dataset, episode_index)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            sampler=episode_sampler,
        )
        print(f"  Frames: {len(episode_sampler)}")
        
        step = 0
        iter_dataloader = iter(dataloader)
        PnPEnv.reset()
        
        while PnPEnv.env.is_viewer_alive():
            PnPEnv.step_env()
            
            if PnPEnv.env.loop_every(HZ=20):
                # Get the action from dataset
                data = next(iter_dataloader)
                
                if step == 0:
                    # Reset object pose based on dataset
                    PnPEnv.set_obj_pose(
                        data['obj_init'][0, :3], 
                        data['obj_init'][0, 3:]
                    )
                
                # Get action and step environment
                action = data['action'].numpy()
                obs = PnPEnv.step(action[0])

                # Visualize images from dataset
                PnPEnv.rgb_agent = data['observation.image'][0].numpy() * 255
                PnPEnv.rgb_ego = data['observation.wrist_image'][0].numpy() * 255
                PnPEnv.rgb_agent = PnPEnv.rgb_agent.astype(np.uint8)
                PnPEnv.rgb_ego = PnPEnv.rgb_ego.astype(np.uint8)
                
                # Transpose from CHW to HWC
                PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1, 2, 0))
                PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1, 2, 0))
                PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
                
                PnPEnv.render()
                step += 1
                print(f"  Frame {step:04d}/{len(episode_sampler)}", end='\r')

                # Move to next episode when current one ends
                if step == len(episode_sampler):
                    print(f"  Frame {step:04d}/{len(episode_sampler)} ✓")
                    break

except KeyboardInterrupt:
    print("\n\nVisualization interrupted by user.")

# Cleanup
print("\nClosing viewer...")
PnPEnv.env.close_viewer()

# Save stats
print("Saving dataset statistics...")
stats = dataset.meta.stats
PATH = dataset.root / 'meta' / 'stats.json'
stats = serialize_dict(stats)
write_json(stats, PATH)

print("\n" + "=" * 60)
print("✓ Visualization complete!")
print("=" * 60)
