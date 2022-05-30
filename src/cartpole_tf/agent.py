import numpy as np
import tensorflow as tf
import keras as ks
from keras.optimizers import Adam
import tensorflow_probability as tfp
from ppo_memory import PPOMemory
from networks import ActorNetwork, CriticNetwork
class Agent:
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
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.memory = PPOMemory(batch_size)
       
    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt + "actor")
        self.critic.save(self.chkpt + "critic")

    def load_models(self):
        print('... loading models ...')
        self.actor = ks.models.load_model(self.chkpt + "actor")
        self.critic = ks.models.load_model(self.chkpt + "critic")

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)

        dist = tfp.distributions.Categorical(probs)
        # print(np.array(dist))
        action = dist.sample()
        # print(action.numpy()[0])
        log_probs = dist.log_prob(action)
        # print(log_probs.numpy()[0])
        value = self.critic(state)

        action = action.numpy()[0]
        value = value.numpy()[0]
        log_probs = log_probs.numpy()[0]

        return action, log_probs, value

    def learn(self):
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
                    states = tf.convert_to_tensor(state_arr[batch])
                    old_probs = tf.convert_to_tensor(old_prob_arr[batch])
                    actions = tf.convert_to_tensor(action_arr[batch])

                    probs = self.actor(states)
                    dist = tfp.distributions.Categorical(probs)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)

                    critic_value = tf.squeeze(critic_value)

                    prob_ratio = tf.math.exp(new_probs - old_probs)

                    weighted_probs = advantage[batch] * prob_ratio
                    clipped_probs = tf.clip_by_value(prob_ratio, 
                                                     1-self.policy_clip,
                                                    1+self.policy_clip)*advantage[batch]
                    weighted_probs = clipped_probs * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage[batch] + values[batch]
                    critic_loss = ks.losses.MSE(returns, critic_value)
                    
                actor_params = self.actor.trainable_variables
                critic_params = self.critic.trainable_variables
                actor_grad = tape.gradient(actor_loss, actor_params)
                critic_grad = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_grad, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grad, critic_params))

        self.memory.clear_memory()  