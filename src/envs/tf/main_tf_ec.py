
import gym
import numpy as np
from sympy import false
from ai import AI
from WheeledRobots import WheeledRobots
from auv_mpc import KineticModel
from utils import plot_learning_curve, plot_learning_curve_2
from time import sleep

if __name__ == '__main__':

    n_games = 500
    agents_amount = 3
    env = WheeledRobots(agents_amount, show=False)

    N = 20
    batch_size = 5
    n_epochs = 1
    alpha = 0.0002

    agents_ai = []
    # print(env.get_action_dim()[0])
    for i in range(agents_amount):
        agents_ai.append(AI(n_actions=8, batch_size=batch_size, 
                        alpha=alpha, n_epochs=n_epochs, 
                        input_dims=env.observation_space.shape))
    env.world.assign_ai(agents_ai)
    
    # agent.load_models()


    figure_file = 'plots/multi/multiple_com_catch/mpc_nn.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    # kinetic_model = KineticModel()
    positions = []

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
            env.render()
            actions = []
            probs = []
            vals = []
            ts = []

            messages = []
            com_probs = []
            com_vals = []
            com_ts = []

            
            for agent in env.world.agents:
                a, t, p, v = agent.choose_action(observation)
                m, com_t, com_p, com_v = agent.gen_message(observation)

                actions.append(a)
                probs.append(p)
                vals.append(v)
                ts.append(t)

                messages.append(m)
                com_probs.append(p)
                com_vals.append(v)
                com_ts.append(t)
            # print(actions)
            observation_, reward, done, info = env.step(actions)
            # print("hehe")
            n_steps += 1
            score += reward

            for agent, a, t, p, v in zip(env.world.agents, actions, ts, probs, vals):
                agent.ai.store_transition(observation, t, p, v, reward, done)
                if n_steps % N == 0:
                    agent.ai.learn()
                    learn_iters += 1

            for agent, m, com_t, com_p, com_v in zip(env.world.agents, messages, com_ts, com_probs, com_vals):
                agent.ai.store_com_transition(observation, com_t, com_p, com_v, reward, done)
                if n_steps % N == 0:
                    agent.ai.com_learn()
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
    x_plots = [i+1 for i in range(n_steps)]
    plot_learning_curve(x, score_history, figure_file, 'Running average of previous ' + str(n_games) + ' games', avg = True)
    # desired = []
    # positions = []
    # hat_positions = []
    # names = []
    # agents_distances = []
    # for a in env.world.agents:
    #     names.append(a.name)
    #     desired.append(a.desired)
    #     positions.append(a.real_position_memory)
    #     hat_positions.append(a.position_memory)
    #     agents_distances.append(a.distances)
    

    # for name, desir, posit, dist in zip(names, desired, positions, agents_distances):
    #     # print(np.array(posit).shape)
    #     posit = np.transpose(posit, (1, 0))
    #     title = '' + 'plots/' + name
    #     desir = np.array(desir)
    #     desir = np.transpose(desir, (1, 2, 0))
        # shape = np.array(hat_pos).shape
        # hat_pos = np.transpose(np.array(hat_pos).reshape(shape[0], shape[1]), (1, 0))
        # print(dist)

        # x_minus_xd = []
        # y_minus_yd = []
        # average_overall_x = []
        # average_overall_y = []
        # tmp = 0
        # average_distances = np.zeros(env.episodes_number)
        # for i in range(n_games):
        #     # print(i)
        #     for j in range(env.episodes_number):
        #         average_distances[j] += dist[i*n_games + j] / n_games
                # print("__________________", (i*n_games+(j+1),"\n")
                # print("\n", tmp)
                # tmp += 1
                # print(posit[0][i*n_games + j], desir[0][0][i*n_games+j])
                # print("abs, ", abs(posit[0][i*n_games + j] - desir[0][0][i*n_games+j]))
                # x_minus_xd.append(abs(posit[0][i*n_games + j] - desir[0][0][i*n_games+j]))
                # y_minus_yd.append(abs(posit[1][i*n_games+j] - desir[1][1][i*n_games + j]))
        # print(x_minus_xd)
        # avg = False
        # xx = [i+1 for i in range(env.episodes_number)]
        # plot_learning_curve_2(x_plots, posit[0], "X", desir[0][0], "Goal", title + "_position_x", "X coordinate")
        # plot_learning_curve_2(x_plots, posit[1], "Y", desir[1][1], "Goal", title + "_position_y", "Y coordinate")
        # plot_learning_curve(x_plots, dist, title + "_ovelall_x", "|x - x_d|", avg = avg)
        # plot_learning_curve(x_plots, dist, title + "_distances", "Distance", avg = avg)
        # plot_learning_curve(xx, average_distances, title + "_average_distances", "Average distance", avg = avg)

        # tmpx = np.zeros((env.episodes_number, n_games))
        # tmpy = np.zeros((env.episodes_number, n_games))
        # for j in range(env.episodes_number):
        #     for i in range(n_games):
        #         tmpx[i][:] = x_minus_xd[j*env.episodes_number : env.episodes_number*(i+1)]
        #         tmpy[i][:] = y_minus_yd[i*env.episodes_number : env.episodes_number*(i+1)]
        # summed_x = np.zeros(env.episodes_number)
        # summed_y = np.zeros(env.episodes_number)
        # for i in range(n_games):
        #     summed_x[i] += tmpx[i]
        #     summed_y[i] += tmpy[i]

        

        # plot_learning_curve(xx, summed_x, title + "_average_error_x", "Average |x - x_d| for" + str(env.episodes_number) + "games", avg = False)
        # plot_learning_curve(xx, summed_y, title + "_average_error_y", "Average |y - y_d| for" + str(env.episodes_number) + "games", avg = False)
        # x_average = [i+1 for i in range(env.episodes_number)]

        # убрать аверраге и вывести то же только с авг етру
        # plot_learning_curve(x_average, average_overall_x, title + "_average_ovelall_x", "Average |x - x_d|", avg = avg)
        # plot_learning_curve(x_average, average_overall_y, title + "_average_overall_y", "Average |y - y_d|", avg = avg)
        # plot_learning_curve_2(x_plots, posit[0], "X", desir[0][0], "Goal", title + "_position_x", "X coordinate")
        # plot_learning_curve_2(x_plots, posit[1], "Y", desir[1][1], "Goal", title + "_position_y", "Y coordinate")
        # plot_learning_curve(x_plots, posit[0], title + "_clear_position_x", "X positions")
        # plot_learning_curve(x_plots, posit[1], title + "_clear_position_y", "X positions")
        # plot_learning_curve(x_plots, hat_pos, title, "hat_positions")

    # print('Turn: [', env.min_turn, ', ', env.max_turn, ']')
    # print('Velo: [', env.min_velo,  ', ', env.max_velo, ']')
    # print('Signal: [', env.min_signal, ', ',  env.max_signal, ']')

    # best_score = env.reward_range[0]
    # score_history = []

    # learn_iters = 0
    # avg_score = 0
    # n_steps = 0
    # # kinetic_model = KineticModel()
    # positions = []

    # loaded_ppo = False
    # observation = env.reset()
    # # if not loaded_ppo:
    # #     for a, ai in zip(env.world.agents, agents_ai):
    # #         ai.load_models(a.name)
    # #     loaded_ppo = True
    # done = False
    # score = 0

    # for a in env.world.agents:
    #     a.desired = []
    #     a.real_position_memory = []
    #     a.position_memory = []
    # while not done:
    #     env.render()
    #     actions = []
    #     probs = []
    #     vals = []
    #     ts = []

    #     # sleep(5)
        
    #     for agent in env.world.agents:
    #         a, t, p, v = agent.choose_action(observation)
    #         actions.append(a)
    #         # vels = kinetic_model.get_velocities([0, 0, a_image.get_orientation()], a)
    #         # pos = kinetic_model.get_position()
    #         # pos = a_image.get_orientation() + pos 
    #         # vel_z = np.sqrt(vels[0]**2 + vels[1]**2)
    #         probs.append(p)
    #         vals.append(v)
    #         ts.append(t)
    #     # print(actions)
    #     observation_, reward, done, info = env.step(actions)
    #     # print("hehe")
    #     n_steps += 1
    #     score += reward

    #     for agent, a, t, p, v in zip(env.world.agents, actions, ts, probs, vals):
    #         agent.ai.store_transition(observation, t, p, v, reward, done)
    #         if n_steps % N == 0:
    #             agent.ai.learn()
    #             learn_iters += 1
    #     observation = observation_

    # score_history.append(score)
    # avg_score = np.mean(score_history[-100:])

    # if avg_score > best_score:
    #     best_score = avg_score
    #     for ai, agent in zip(agents_ai, env.world.agents):
    #         ai.save_models(agent.name)

    # print('episode', 1, 'score %.1f' % score, 'avg score %.1f' % avg_score,
    #             'time_steps', n_steps, 'learning_steps', learn_iters)

    # # x = [i+1 for i in range(len(score_history))]
    # x_plots = [i+1 for i in range(n_steps)]
    # # plot_learning_curve(x, score_history, figure_file, 'Running average of previous ' + str(n_games) + ' scores')
    # desired = []
    # positions = []
    # hat_positions = []
    # names = []
    # for a in env.world.agents:
    #     names.append(a.name)
    #     desired.append(a.desired)
    #     positions.append(a.real_position_memory)
    #     hat_positions.append(a.position_memory)
    

    # for name, desir, posit, hat_pos in zip(names, desired, positions, hat_positions):
    #     # print(np.array(posit).shape)
    #     posit = np.transpose(posit, (1, 0))
    #     title = '' + 'plots_thesis_final/' + name
    #     desir = np.array(desir)
    #     desir = np.transpose(desir, (1, 2, 0))
    #     shape = np.array(hat_pos).shape
    #     hat_pos = np.transpose(np.array(hat_pos).reshape(shape[0], shape[1]), (1, 0))
    #     u = env.ep_return
    #     x = []
    #     y = []
    #     average_overall_x = []
    #     average_overall_y = []
    #     for j in range(env.episodes_number):
    #         # print("___", j)
    #         # print((i)*env.episodes_number + j)
    #         # print(abs(posit[0][(i)*env.episodes_number + j] - desir[0][0][(i)*env.episodes_number + j]))
    #         x.append(abs(posit[0][j] - desir[0][0][j]))
    #         y.append(abs(posit[1][j] - desir[1][1][j]))

    #     for j in range(env.episodes_number):

    #         average_overall_x.append(0.0)
    #         average_overall_y.append(0.0)
            
    #         average_overall_x[j] += x[j]
    #         average_overall_y[j] += y[j]
            
    #     # print(x)
    #     # print(desir.shape)
    #     # print(desir[0][0], hat_pos[0][0])

    #     x_average = [i+1 for i in range(len(average_overall_x))]
    #     plot_learning_curve(x_plots, x, title + "_ovelall_x", "|x - x_d|")
    #     plot_learning_curve(x_plots, y, title + "_overall_y", "|y - y_d|")
    #     plot_learning_curve_2(x_plots, posit[0], "X", desir[0][0], "Goal", title + "_position_x", "X coordinate")
    #     plot_learning_curve_2(x_plots, posit[1], "Y", desir[1][1], "Goal", title + "_position_y", "Y coordinate")
    env.close()