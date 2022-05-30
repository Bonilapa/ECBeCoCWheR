import gym
import numpy as np
from ai import AI
from WheeledRobots import WheeledRobots
from utils import plot_learning_curve

if __name__ == '__main__':
    agents_amount = 2
    env = WheeledRobots(agents_amount)

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agents_ai = []
    # print(env.get_action_dim()[0])
    for i in range(agents_amount):
        agents_ai.append(AI(n_actions=env.get_action_dim()[0], batch_size=batch_size, 
                        alpha=alpha, n_epochs=n_epochs, 
                        input_dims=env.observation_space.shape))
    
    # agent.load_models()

    n_games = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0


    loaded_ppo = False
    for i in range(n_games):
        observation = env.reset()
        # if not loaded_ppo:
        #     for a, ai in zip(env.world.agents, agents_ai):
        #         ai.load_models(a.name)
        #     loaded_ppo = True
        done = False
        score = 0
        while not done:
            # env.render()
            actions = []
            probs = []
            vals = []

            for ai in agents_ai:
                a, p, v = ai.choose_action(observation)
                actions.append(a)
                probs.append(p)
                vals.append(v)

            observation_, reward, done, info = env.step(actions)
            n_steps += 1
            score += reward

            for ai, a, p, v in zip(agents_ai, actions, probs, vals):
                ai.store_transition(observation, a, p, v, reward, done)
                if n_steps % N == 0:
                    ai.learn()
                    learn_iters += 1
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            for ai, agent in zip(agents_ai, env.world.agents):
                ai.save_models(agent.name)

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
    
env.close()