import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
import click
import torch
from src.monitor import run_agent, test_agent
from src.agent import Agent

# path to Reacher UnityEnvironment
# i.e. 'Reacher.app'
REACHER_APP = ''


def plot_scores(agent, scores, rolling_window=10, save_fig=True):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'{str(agent)} scores')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean);

    if save_fig:
        plt.savefig(f'figures/{str(agent)}_scores.png',
                    bbox_inches='tight', pad_inches=0)

    return rolling_mean


@click.command()
@click.option('--test', help='test or train agent', is_flag=True)
def main(test):
    # init the environment
    env = UnityEnvironment(file_name=REACHER_APP)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of actions
    action_size = brain.vector_action_space_size
    # dimenison of the state space
    state_size = env_info.vector_observations.shape[1]
    # number of agents
    n_agents = len(env_info.agents)

    # create a DDPG agent
    agent = Agent(state_size=state_size,
                  action_size=action_size,
                  n_agents=n_agents,
                  random_seed=1)
    if not test:
        # train the agent
        scores = run_agent(env, agent, n_episodes=300)
        _ = plot_scores(agent, scores)
    else:
        # test the trained agent
        # load the weights from file
        agent.actor_local.load_state_dict(
            torch.load(f'weights/{str(agent)}_checkpoint_actor.pth'))
        agent.critic_local.load_state_dict(
            torch.load(f'weights/{str(agent)}_checkpoint_critic.pth'))
        test_agent(env, agent, n_agents)

    env.close()


if __name__ == '__main__':
    main()
