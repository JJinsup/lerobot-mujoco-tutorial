#!/usr/bin/env python3
"""
Standalone script to deploy trained ACT policy
Run from terminal: python 4.deploy_standalone.py
"""
import os
import torch
import numpy as np
import torchvision
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import write_json, serialize_dict, dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from mujoco_env.y_env import SimpleEnv

print("=" * 60)
print("Deploying Trained ACT Policy")
print("=" * 60)

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
POLICY_PATH = './ckpt/act_y'
DATASET_ROOT = './demo_data'
DATASET_REPO = 'omy_pnp'
XML_PATH = './asset/example_scene_y.xml'
TASK_NAME = 'Put mug cup on the plate'
SEED = 0

print(f"\nUsing device: {DEVICE}")

# Load dataset metadata
print("\nLoading dataset metadata...")
try:
    dataset_metadata = LeRobotDatasetMetadata(DATASET_REPO, root=DATASET_ROOT)
    print("✓ Dataset metadata loaded")
except Exception as e:
    print(f"✗ Error loading dataset metadata: {e}")
    exit(1)

# Setup features
print("Setting up policy features...")
features = dataset_to_policy_features(dataset_metadata.features)
output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
input_features = {key: ft for key, ft in features.items() if key not in output_features}
input_features.pop("observation.wrist_image", None)
print(f"✓ Input features: {list(input_features.keys())}")
print(f"✓ Output features: {list(output_features.keys())}")

# Load policy
print(f"\nLoading policy from {POLICY_PATH}...")
try:
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=1,
        temporal_ensemble_coeff=0.9
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    policy = ACTPolicy.from_pretrained(
        POLICY_PATH,
        config=cfg,
        dataset_stats=dataset_metadata.stats
    )
    policy.to(DEVICE)
    policy.eval()
    print("✓ Policy loaded successfully")
except Exception as e:
    print(f"✗ Error loading policy: {e}")
    print(f"  Make sure checkpoint exists at {POLICY_PATH}")
    exit(1)

# Load environment
print(f"\nInitializing environment...")
try:
    PnPEnv = SimpleEnv(XML_PATH, action_type='joint_angle', init_viewer=True)
    print("✓ Environment initialized")
except Exception as e:
    print(f"✗ Error initializing environment: {e}")
    exit(1)

# Load dataset to get object positions from training data
print(f"\nLoading dataset for object position reference...")
try:
    dataset = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT)
    # Get first sample to extract object positions from training
    first_sample = dataset[0]
    obj_init_pose = first_sample['obj_init'].numpy()
    print(f"✓ Object positions from training data: {obj_init_pose}")
except Exception as e:
    print(f"Warning: Could not load object positions from dataset: {e}")
    obj_init_pose = None

print("\n" + "=" * 60)
print("Running policy rollout...")
print("=" * 60 + "\n")

# Setup
step = 0
success_count = 0
max_steps = 500  # Safety limit to prevent infinite loops
img_transform = torchvision.transforms.ToTensor()
first_reset = True

try:
    while PnPEnv.env.is_viewer_alive() and step < max_steps:
        PnPEnv.step_env()
        
        if PnPEnv.env.loop_every(HZ=20):
            # On first reset, set object positions from training data
            if first_reset and obj_init_pose is not None:
                # Extract mug and plate positions from obj_init_pose
                mug_xyz = obj_init_pose[:3]
                plate_xyz = obj_init_pose[3:]
                
                # Set object poses to match training data
                PnPEnv.set_obj_pose(mug_xyz, np.array([0, 0, 0]))  # mug
                obj_names = PnPEnv.env.get_body_names(prefix='body_obj_')
                if len(obj_names) > 1:
                    PnPEnv.env.set_p_base_body(body_name=obj_names[1], p=plate_xyz)
                    PnPEnv.env.set_R_base_body(body_name=obj_names[1], R=np.eye(3))
                PnPEnv.env.forward(increase_tick=False)
                first_reset = False
                print(f"✓ Object positions matched to training data")
            
            # Check if the task is completed
            success = PnPEnv.check_success()
            if success:
                success_count += 1
                print(f"\n✓ Success #{success_count}!")
                
                # Reset for next rollout
                policy.reset()
                PnPEnv.reset(seed=SEED)
                first_reset = True  # Set to reapply object positions on next iteration
                step = 0
                continue
            
            # Get the current state
            state = PnPEnv.get_ee_pose()
            
            # Get images from environment
            agent_image, wrist_image = PnPEnv.grab_image()
            
            # Process agent image
            agent_image = Image.fromarray(agent_image)
            agent_image = agent_image.resize((256, 256))
            agent_image = img_transform(agent_image)
            
            # Process wrist image
            wrist_image = Image.fromarray(wrist_image)
            wrist_image = wrist_image.resize((256, 256))
            wrist_image = img_transform(wrist_image)
            
            # Prepare input data for policy
            data = {
                'observation.state': torch.tensor([state]).to(DEVICE),
                'observation.image': agent_image.unsqueeze(0).to(DEVICE),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(DEVICE),
                'task': [TASK_NAME],
                'timestamp': torch.tensor([step / 20.0]).to(DEVICE)
            }
            
            # Select action from policy
            with torch.no_grad():
                action = policy.select_action(data)
            action = action[0].cpu().detach().numpy()
            
            # Execute action
            _ = PnPEnv.step(action)
            PnPEnv.render()
            
            step += 1
            if step % 20 == 0:
                print(f"Step: {step:04d}", end='\r')

except KeyboardInterrupt:
    print("\n\nRollout interrupted by user.")

# Cleanup
print("\n\nClosing viewer...")
PnPEnv.env.close_viewer()

print("\n" + "=" * 60)
print("✓ Policy deployment complete!")
print(f"Total successes: {success_count}")
print("=" * 60)
