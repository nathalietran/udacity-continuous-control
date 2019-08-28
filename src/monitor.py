from collections import deque
import torch
import numpy as np

def run_agent(env, agent, n_episodes=1000, max_t=1000, print_every=100):
    """Run the agent inside an environment.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        # get the current state (for each agent)
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(len(env_info.agents))
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if any(dones):
                break
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))

        print('\rEpisode {}\tAverage Score: {:.2f}'
              .format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                  .format(i_episode - print_every, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(),
                       f'weights/{str(agent)}_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(),
                       f'weights/{str(agent)}_checkpoint_critic.pth')
            break
    return scores


def test_agent(env, agent, n_agents, max_t=1000):
    """Test the trained agent

    Params
    ======
        max_t (int): maximum number of timesteps per episode
    """
    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(n_agents)
    for t in range(max_t):
        actions = agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    print('Total score (averaged over agents) this episode: {}'
          .format(np.mean(scores)))
