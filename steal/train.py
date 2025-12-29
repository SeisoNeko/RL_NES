"""
Main script for training and testing Tetris PPO agent
"""

from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT
from prepare_feature import crop_clean_state, extra_feats
from controller import Controller, TestController, PPOAgent, compute_reward
import tensorflow as tf
import numpy as np
import time
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Remove to regain GPU ability
tf.compat.v1.disable_eager_execution()

# Number of possible actions that PPO Agent can take
act_space_size = 6


def train(agent, epochs, batch_steps, episode_steps):
    """
    Trains PPO agent with specified number of epochs, batch_steps, and episode_steps
    """
    final_out = ""
    lll = []

    # Use agent's controller as the main environment controller
    controller = agent.controller
    env = agent.env
    done = True

    # Epoch loop
    for epoch in range(epochs):
        print(epoch)
        st = time.perf_counter()
        ll = []

        # Save checkpoints
        if epoch % 10 == 1:
            save_model(agent, str(epoch))

        # Data collection loop (before model refitting)
        while len(controller.X1) < batch_steps:
            print(len(controller.X1))
            # Reset the environment
            if done:
                obs = env.reset()
                env.render()  # <-------------------remove to avoid render

            # Get raw observation and create new observation state
            raw_obs = obs
            cleaned_obs = crop_clean_state(raw_obs)
            info_vec = np.zeros(shape=(1, 5))
            prev_holes = 0
            prev_bumps = 0

            # Loop through episodes/games
            rews = []
            steps = 0
            while True:
                steps += 1

                # Prediction, action, save prediction
                agent_pred, agent_act = [
                    x[0] for x in agent.controller.pf([cleaned_obs[None], info_vec])
                ]
                agent.controller.P.append(agent_pred)

                # Add a decaying randomness and user input (disabled w/ -1) to the chosen action
                probability = 1 - (10 * epoch) / epochs
                probability = 0 if probability < 0 else probability
                if np.random.random_sample() < probability:
                    if epoch == -1:
                        event = input(f"Action {steps}: ")
                        if event == "s":
                            agent_act = 5
                        elif event == "a":
                            agent_act = 4
                        elif event == "d":
                            agent_act = 3
                        elif event == "q":
                            agent_act = 2
                        elif event == "e":
                            agent_act = 1
                        else:
                            agent_act = 0
                    else:
                        agent_act = np.random.choice(act_space_size)

                # Save this state action pair
                agent.save_pair(cleaned_obs, info_vec, agent_act)

                # Take the action and save the reward
                obs, agent_rew, done, info = env.step(agent_act)
                env.render()  # <-------------------remove to avoid render
                raw_obs = obs
                cleaned_obs = crop_clean_state(raw_obs)
                agent_rew, prev_holes, prev_bumps = compute_reward(
                    cleaned_obs, agent_rew, prev_holes, prev_bumps
                )
                # Take bonus steps to simplify:
                if not done:
                    obs, fake_rew, done, info = env.step(5)
                    env.render()  # <-------------------remove to avoid render
                    raw_obs = obs
                    cleaned_obs = crop_clean_state(raw_obs)
                    fake_rew, prev_holes, prev_bumps = compute_reward(
                        cleaned_obs, fake_rew, prev_holes, prev_bumps
                    )
                    agent_rew += fake_rew
                    agent_rew /= 2

                # print(agent_rew)
                info_vec = extra_feats(info)

                rews.append(agent_rew)

                if steps == episode_steps or done:
                    ll.append(np.sum(rews))

                    for i in range(len(rews) - 2, -1, -1):
                        rews[i] += rews[i + 1] * agent.controller.gamma
                    agent.controller.V.extend(rews)

                    break

        # Model fitting
        loss, vloss = agent.controller.fit()
        # loss, vloss = (None, None)
        # controller.X1 = []

        if loss != None and vloss != None:
            lll.append(
                (
                    epoch,
                    np.mean(ll),
                    loss,
                    vloss,
                    len(agent.controller.X1),
                    len(ll),
                    time.perf_counter() - st,
                )
            )
            print(
                "%3d  ep_rew:%9.2f  loss:%7.2f   vloss:%9.2f  counts: %5d/%3d tm: %.2f s"
                % lll[-1]
            )
            print("Episode No: %3d  Episode Reward: %9.2f" % (lll[-1][0], lll[-1][1]))
            sign = "+" if lll[-1][1] >= 0 else ""
            final_out += sign + str(lll[-1][1])


def test(agent, episode_steps):
    """
    Tests PPO agent (plays game) with specified number of episode_steps
    """

    env = agent.env

    # Reset the environment
    obs = env.reset()
    env.render()  # <-------------------remove to avoid render

    # Get raw observation and create new observation state
    raw_obs = obs
    cleaned_obs = crop_clean_state(raw_obs)
    info_vec = np.zeros(shape=(1, 5))
    prev_holes = 0
    prev_bumps = 0

    # Loop through episodes/game
    steps = 0
    while True:
        steps += 1

        # Prediction, action
        _, agent_act = [
            x[0] for x in agent.controller.pf([cleaned_obs[None], info_vec])
        ]

        # Take the action
        obs, agent_rew, done, info = env.step(agent_act)
        env.render()  # <-------------------remove to avoid render
        raw_obs = obs
        cleaned_obs = crop_clean_state(raw_obs)
        agent_rew, prev_holes, prev_bumps = compute_reward(
            cleaned_obs, agent_rew, prev_holes, prev_bumps
        )
        # Take bonus steps to simplify:
        if not done:
            obs, fake_rew, done, info = env.step(5)
            env.render()  # <-------------------remove to avoid render
            raw_obs = obs
            cleaned_obs = crop_clean_state(raw_obs)
            fake_rew, prev_holes, prev_bumps = compute_reward(
                cleaned_obs, fake_rew, prev_holes, prev_bumps
            )
            agent_rew += fake_rew
            agent_rew /= 2

        info_vec = extra_feats(info)

        if steps == episode_steps or done:
            break


def save_model(agent, name):
    """
    Saves the PPO agent models to the saved_models dir using the given name
    """
    print()
    print("saving popt")
    tf.keras.models.save_model(agent.controller.popt, f"./saved_models/popt_{name}")
    print("saving p")
    tf.keras.models.save_model(agent.controller.p, f"./saved_models/p_{name}")
    print("saving v")
    tf.keras.models.save_model(agent.controller.v, f"./saved_models/v_{name}")


def main():
    """
    Main function to kick of training/testing.
    Can adjust agent name, loaded models, and hyperparameters here.
    """
    # Hyperparams
    gamma = 0.99
    epochs = 100
    batch_steps = 1500
    episode_steps = 1500

    # Train/test toggle
    train_system = False

    # Create env
    env = gym_tetris.make("TetrisA-v2")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Declare observation shape, action space
    observation_shape = (160, 80, 1)
    action_space = env.action_space

    if train_system:
        # Train
        controller = Controller(gamma, observation_shape, action_space, "CONTROLLER")
        agent = PPOAgent(env, controller=controller)
        train(
            agent, epochs=epochs, batch_steps=batch_steps, episode_steps=episode_steps
        )
        save_model(agent, "trial")
    else:
        # Test
        v = tf.keras.models.load_model("./saved_models/v_trial", compile=False)
        p = tf.keras.models.load_model("./saved_models/p_trial", compile=False)
        popt = tf.keras.models.load_model("./saved_models/popt_trial", compile=False)
        controller = TestController(
            gamma, observation_shape, action_space, "CONTROLLER", v, p, popt
        )
        agent = PPOAgent(env, controller=controller)
        test(agent, episode_steps)
    env.close()


if __name__ == "__main__":
    main()