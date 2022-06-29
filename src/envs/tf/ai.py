import numpy as np
import tensorflow as tf
import keras as ks
from keras.optimizers import Adam
import tensorflow_probability as tfp
# from auv_mpc import KineticModel
from ppo_memory import PPOMemory
from networks import ActorNetwork, CriticNetwork
class AI:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003,
            gae_lambda=0.95, policy_clip=0.2, batch_size=64,
            n_epochs=10, chkpt = 'models/'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt = chkpt

        self.actor = ActorNetwork(n_actions)
        self.critic = CriticNetwork()
        self.com_actor = ActorNetwork(n_actions)
        self.com_critic = CriticNetwork()
        # self.kinetic_model = KineticModel()
        
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.com_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.com_critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)
        self.com_memory = PPOMemory(batch_size)
       
    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    def store_com_transition(self, state, message, com_probs, com_vals, reward, done):
        self.com_memory.store_memory(state, message, com_probs, com_vals, reward, done)

    def save_models(self, name):
        print('... saving models ...', name)
        self.actor.save(self.chkpt + "actor_" + name)
        self.critic.save(self.chkpt + "critic_" + name)
        self.com_actor.save(self.chkpt + "com_actor_" + name)
        self.com_critic.save(self.chkpt + "com_critic_" + name)

    def load_models(self, name):
        print('... loading models ...', name)
        self.actor = ks.models.load_model(self.chkpt + "actor_"+ name)
        self.critic = ks.models.load_model(self.chkpt + "critic_"+name)
        self.com_actor = ks.models.load_model(self.chkpt + "com_actor_"+ name)
        self.com_critic = ks.models.load_model(self.chkpt + "com_critic_"+name)

    def gen_message(self, observation):
        observation = [np.float16(p) for p in observation]
        # print(observation)
        state = tf.convert_to_tensor([observation])
        # print("\nState:\n", state)
        probs = self.actor(state)
        # print("\nprobs\n", probs, "\n")

        action = probs.numpy()[0]
        
        dist = tfp.distributions.Categorical(probs)
        # print("\ndist:\n", dist)
        tmp = dist.sample()
        val = tmp.numpy()[0]
        if val == 8:
            val -= 1
            tmp = tf.convert_to_tensor(val)
        # print("\naction:\n",action)
        # print('tmp: \n', tmp)

        # pr = 0.1*action[0] + 0.01*action[1]  + 0.001*action[2]
        # print('pr: \n', pr)
        log_probs = dist.log_prob(tmp)
        # print('log_probs:\n', log_probs)
        value = self.critic(state)
        # action = action.numpy()[0]
        value = value.numpy()[0]
        log_probs = log_probs.numpy()[0]

        return action, tmp, log_probs, value

    def choose_action(self, observation):
        observation = [np.float16(p) for p in observation]
        # print(observation)
        state = tf.convert_to_tensor([observation])
        # print("\nState:\n", state)
        probs = self.actor(state)
        # print("\nprobs\n", probs, "\n")

        action = probs.numpy()[0]
        
        dist = tfp.distributions.Categorical(probs)
        # print("\ndist:\n", dist)
        tmp = dist.sample()
        val = tmp.numpy()[0]
        if val == 8:
            val -= 1
            tmp = tf.convert_to_tensor(val)
        # print("\naction:\n",action)
        # print('tmp: \n', tmp)

        # pr = 0.1*action[0] + 0.01*action[1]  + 0.001*action[2]
        # print('pr: \n', pr)
        log_probs = dist.log_prob(tmp)
        # print('log_probs:\n', log_probs)
        value = self.critic(state)
        # action = action.numpy()[0]
        value = value.numpy()[0]
        log_probs = log_probs.numpy()[0]

        return action, tmp, log_probs, value

    def learn(self):
        # print("\n______learn_________\n")
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:

                    # state_arr[batch] = 
                    states = tf.convert_to_tensor([np.float16(p) for p in state_arr[batch]])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    # print(actions)
                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)

                    # print(actions)
                    # actions = actions.numpy().tolist()
                    # print(actions)
                    # act = []
                    # for a in actions:
                    #     act.append(0.1*a[0] + 0.01*a[1] + 0.001*a[2])
                    new_probs = dist.log_prob(actions)
                    # new_probs = tf.squeeze(tf.convert_to_tensor(act))
                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value)

                    prob_ratio = tf.math.exp(new_probs - old_probs)

                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 
                                                     1-self.policy_clip,
                                                    1+self.policy_clip)*advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = ks.losses.MSE(returns, critic_value)
                    
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables
                # lena = len(actor_params)
                # lenc = len(critic_params)
                # print("\n Actor: ", actor_params)
                # print("\n Critic: ", critic_params, "\n")
                # print("\n------------\n actor: ", lena, "Critic: ", lenc)
                # for i,j in zip(range(lena), range(lenc)):
                #     print("\n Actor: ", actor_params[i])
                #     print("\n Critic: ", critic_params[j], "\n")
                actor_grad = tape.gradient(actor_loss, actor_params)
                critic_grad = tape.gradient(critic_loss, critic_params)
                # print("\n grads\n", actor_grad, "\n")
                self.actor.optimizer.apply_gradients(zip(actor_grad, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grad, critic_params))

        self.memory.clear_memory()     

    def com_learn(self):
        # print("\n______learn_________\n")
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.com_memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t

            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:

                    states = tf.convert_to_tensor([np.float16(p) for p in state_arr[batch]])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.com_actor(states)
                    dist = tfp.distributions.Categorical(probs)

                    new_probs = dist.log_prob(actions)

                    com_critic_value = self.com_critic(states)

                    com_critic_value = tf.squeeze(com_critic_value)

                    prob_ratio = tf.math.exp(new_probs - old_probs)

                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 
                                                     1-self.policy_clip,
                                                    1+self.policy_clip)*advantage[batch]
                    com_actor_loss = -tf.math.minimum(weighted_probs, clipped_probs)
                    com_actor_loss = tf.math.reduce_mean(com_actor_loss)

                    returns = advantage[batch] + values[batch]
                    com_critic_loss = ks.losses.MSE(returns, com_critic_value)
                    
                com_actor_params = self.com_actor.trainable_variables
                com_critic_params = self.com_critic.trainable_variables
                com_actor_grad = tape.gradient(com_actor_loss, com_actor_params)
                com_critic_grad = tape.gradient(com_critic_loss, com_critic_params)
                self.com_actor.optimizer.apply_gradients(zip(com_actor_grad, com_actor_params))
                self.com_critic.optimizer.apply_gradients(zip(com_critic_grad, com_critic_params))

        self.com_memory.clear_memory()  