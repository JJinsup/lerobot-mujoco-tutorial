"""
Deploy trained SmolVLA policy in simulation with mujoco rendering.
This script loads a trained model and runs it in the environment.
"""

import torch
from PIL import Image
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features

from mujoco_env.y_env2 import SimpleEnv2


def get_default_transform(image_size: int = 224):
    """
    Returns a torchvision transform that:
    Converts to a FloatTensor and scales pixel values [0,255] -> [0.0,1.0]
    """
    return transforms.Compose([
        transforms.ToTensor(),  # PIL [0–255] -> FloatTensor [0.0–1.0], shape C×H×W
    ])


def load_policy(device='cuda', checkpoint_path='./ckpt/smolvla_omy/checkpoints/last/pretrained_model'):
    """Load trained SmolVLA policy and dataset metadata."""
    
    # Load dataset metadata
    try:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./demo_data_language')
    except:
        dataset_metadata = LeRobotDatasetMetadata("omy_pnp_language", root='./omy_pnp_language')
    
    # Prepare features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    
    # Create config
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=5,
        n_action_steps=5
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    
    # Load policy
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path, dataset_stats=dataset_metadata.stats)
    policy.to(device)
    policy.eval()
    
    return policy, dataset_metadata


def deploy(
    policy,
    env,
    device='cuda',
    max_steps=None,
    seed=0
):
    """
    Deploy policy in environment with rendering.
    
    Args:
        policy: Loaded SmolVLAPolicy model
        env: Simulation environment
        device: Device to run inference on
        max_steps: Maximum steps per episode (None = no limit)
        seed: Random seed for environment
    """
    
    step = 0
    img_transform = get_default_transform()
    
    env.reset(seed=seed)
    policy.reset()
    
    while env.env.is_viewer_alive():
        env.step_env()
        
        if env.env.loop_every(HZ=20):
            # Check if the task is completed
            success = env.check_success()
            if success:
                print('✓ Task completed successfully!')
                # Reset the environment and action queue
                policy.reset()
                env.reset(seed=seed)
                step = 0
                continue
            
            # Check max steps limit
            if max_steps is not None and step >= max_steps:
                print(f'Reached maximum steps ({max_steps})')
                policy.reset()
                env.reset(seed=seed)
                step = 0
                continue
            
            # Get the current state of the environment
            state = env.get_joint_state()[:6]
            
            # Get the current images from the environment
            image, wrist_image = env.grab_image()
            
            # Process main image
            image = Image.fromarray(image)
            image = image.resize((256, 256))
            image = img_transform(image)
            
            # Process wrist image
            wrist_image = Image.fromarray(wrist_image)
            wrist_image = wrist_image.resize((256, 256))
            wrist_image = img_transform(wrist_image)
            
            # Prepare data for policy
            data = {
                'observation.state': torch.tensor([state]).to(device),
                'observation.image': image.unsqueeze(0).to(device),
                'observation.wrist_image': wrist_image.unsqueeze(0).to(device),
                'task': [env.instruction],
            }
            
            # Select an action
            with torch.no_grad():
                action = policy.select_action(data)
            action = action[0, :7].cpu().detach().numpy()
            
            # Take a step in the environment
            _ = env.step(action)
            env.render()
            
            step += 1


def main():
    """Main deployment function."""
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT_PATH = './ckpt/smolvla_omy/checkpoints/last/pretrained_model'
    XML_PATH = './asset/example_scene_y2.xml'
    
    print("Loading policy...")
    policy, dataset_metadata = load_policy(device=DEVICE, checkpoint_path=CHECKPOINT_PATH)
    
    print("Creating environment...")
    env = SimpleEnv2(XML_PATH, action_type='joint_angle')
    
    print("Deploying policy...")
    print(f"Task instruction: {env.instruction}")
    print("Press Ctrl+C to stop or close the window.")
    
    try:
        deploy(policy, env, device=DEVICE, seed=0)
    except KeyboardInterrupt:
        print("\nDeployment stopped by user.")
    finally:
        print("Deployment finished.")


if __name__ == "__main__":
    main()
