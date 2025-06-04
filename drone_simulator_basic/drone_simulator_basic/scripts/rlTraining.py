import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
import os

# Import custom environment
from drone_env import DroneEnv  # update path as needed

def linear_schedule(initial_value, final_value):
    def schedule(progress_remaining):
        return final_value + progress_remaining * (initial_value - final_value)
    return schedule


def make_env():
    return Monitor(DroneEnv())  # Wrap with Monitor for logging

def main():
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create training environment
    env = make_vec_env(make_env, n_envs=1)

    # Evaluation environment
    eval_env = make_env()

    # Callback to save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "/best_model",
        log_path=log_dir,
        eval_freq=1500,
        deterministic=True,
        render=False
    )

    # Create the model
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(1e-3,1e-5),
        batch_size=256,
        buffer_size=100_000,
        learning_starts=10_000,
        train_freq=(1, "step"),
        tau=0.005,
        gamma=0.99,
        tensorboard_log=log_dir,
    )

    # Train the model
    model.learn(total_timesteps=600_000, callback=eval_callback)

    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

    print("Training complete. Model saved.")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Final evaluation reward: {mean_reward:.2f} ± {std_reward:.2f}")

if __name__ == "__main__":
    main()
