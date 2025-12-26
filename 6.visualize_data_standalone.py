import sys
import os
import argparse
import subprocess
import yaml  # PyYAML 라이브러리 필요
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

# LeRobot 및 MuJoCo 관련 라이브러리 임포트
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from mujoco_env.y_env2 import SimpleEnv2
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the root of the 'lerobot-mujoco-tutorial' directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 1. Helper Classes & Functions
# -----------------------------------------------------------------------------

class EpisodeSampler(Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

def load_config(config_path):
    """YAML 설정 파일을 로드합니다."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at '{config_path}'")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            sys.exit(1)

def download_dataset_if_needed(repo_id, target_dir):
    if os.path.exists(target_dir):
        print(f"Directory '{target_dir}' already exists. Skipping download.")
        return

    print(f"Downloading dataset from {repo_id}...")
    try:
        subprocess.run(["git", "clone", f"https://huggingface.co/datasets/{repo_id}"], check=True)
        print("Download complete.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------------
# 2. Main Logic
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot Dataset using YAML Config")
    
    # 설정 파일 경로
    parser.add_argument("--config", type=str, default="smolvla_omy.yaml", 
                        help="Path to the .yaml configuration file (default: smolvla_omy.yaml)")
    
    # 기능 제어 인자
    parser.add_argument("--start_episode", type=int, default=0, 
                        help="Start playing from this episode index (default: 0)")
    parser.add_argument("--download", action="store_true", 
                        help="Download dataset based on repo_id in YAML")
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Push the dataset to Hugging Face Hub ONLY (Skip visualization)")
    
    args = parser.parse_args()

    # 1. YAML 파일 로드 및 정보 추출
    print(f"Reading configuration from: {args.config}")
    cfg = load_config(args.config)
    
    try:
        repo_id = cfg['dataset']['repo_id']
        root_dir = cfg['dataset']['root']
        print(f" -> Dataset Repo ID: {repo_id}")
        print(f" -> Dataset Root Dir: {root_dir}")
    except KeyError as e:
        print(f"Error: Missing key in YAML file: {e}")
        print("Ensure your YAML has 'dataset' key with 'repo_id' and 'root'.")
        sys.exit(1)

    # 2. 데이터 다운로드 (옵션)
    if args.download:
        repo_name = repo_id.split("/")[-1]
        target = root_dir if root_dir else repo_name
        download_dataset_if_needed(repo_id, target)

    # 3. 데이터셋 로드
    print("Loading dataset...")
    try:
        dataset = LeRobotDataset(repo_id, root=root_dir)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # =========================================================
    # [핵심 변경] 업로드 옵션이 켜져 있으면 시각화 건너뛰고 바로 업로드 수행
    # =========================================================
    if args.push_to_hub:
        print("\n" + "=" * 50)
        print("🚀 UPLOAD MODE DETECTED")
        print("Skipping visualization and pushing directly to Hub...")
        print("=" * 50 + "\n")

        print(f"Pushing dataset to Hugging Face Hub.")
        print(f"Repo ID: {repo_id}")
        print("Note: You must be logged in via 'huggingface-cli login'.")
        
        try:
            dataset.push_to_hub(upload_large_folder=True)
            print("\n✅ Upload completed successfully!")
        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
        
        # 업로드 후 프로그램 종료
        return 

    # ---------------------------------------------------------
    # 아래부터는 시각화(Visualization) 로직 (업로드 모드가 아닐 때만 실행됨)
    # ---------------------------------------------------------

    total_episodes = len(dataset.episode_data_index["from"])
    print(f"Total Episodes found: {total_episodes}")

    # 4. MuJoCo 환경 초기화
    xml_path = './asset/example_scene_y2.xml'
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        sys.exit(1)
        
    print("Initializing MuJoCo environment...")
    pnp_env = SimpleEnv2(xml_path, action_type='joint_angle')

    # 5. 시각화 루프 (Sequential Playback)
    print(f"Starting sequential visualization from Episode {args.start_episode}.")
    print("Close the viewer window to stop.")

    current_episode_idx = args.start_episode
    if current_episode_idx >= total_episodes:
        current_episode_idx = 0

    episode_sampler = EpisodeSampler(dataset, current_episode_idx)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, sampler=episode_sampler)
    iter_dataloader = iter(dataloader)
    
    step = 0
    pnp_env.reset()

    try:
        while pnp_env.env.is_viewer_alive():
            pnp_env.step_env()
            
            if pnp_env.env.loop_every(HZ=20):
                try:
                    data = next(iter_dataloader)
                except StopIteration:
                    print(f"Episode {current_episode_idx} finished.")
                    
                    current_episode_idx += 1
                    if current_episode_idx >= total_episodes:
                        print("Reached end of playlist. Restarting from Episode 0.")
                        current_episode_idx = 0
                    
                    print(f"Now playing Episode {current_episode_idx}...")

                    episode_sampler = EpisodeSampler(dataset, current_episode_idx)
                    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, sampler=episode_sampler)
                    iter_dataloader = iter(dataloader)
                    
                    pnp_env.reset()
                    step = 0
                    data = next(iter_dataloader)

                if step == 0:
                    instruction = data['task'][0]
                    pnp_env.set_instruction(instruction)
                    obj_init = data['obj_init'][0]
                    pnp_env.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])

                action = data['action'].numpy()
                obs = pnp_env.step(action[0])

                rgb_agent = data['observation.image'][0].numpy() * 255
                rgb_ego = data['observation.wrist_image'][0].numpy() * 255
                
                pnp_env.rgb_agent = np.transpose(rgb_agent.astype(np.uint8), (1, 2, 0))
                pnp_env.rgb_ego = np.transpose(rgb_ego.astype(np.uint8), (1, 2, 0))
                pnp_env.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)

                pnp_env.render()
                step += 1
                
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    finally:
        print("Closing viewer...")
        pnp_env.env.close_viewer()

if __name__ == "__main__":
    main()