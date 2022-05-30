
import numpy as np
import torch as T
from ppo_memory import PPOMemory
from networks import ActorNetwork, CriticNetwork
class AI:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor_turn = ActorNetwork(n_actions, input_dims, alpha)
        self.critic_turn = CriticNetwork(input_dims, alpha)
        self.memory_turn = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory_turn.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, name):
        print('... saving models ', name)
        self.actor_turn.save_checkpoint(name)
        self.critic_turn.save_checkpoint(name)

    def load_models(self, name):
        print('... loading models ', name)
        self.actor_turn.load_checkpoint(name)
        self.critic_turn.load_checkpoint(name)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor_turn.device)

        dist_turn = self.actor_turn(state)
        value_turn = self.critic_turn(state)
        # print(dist_turn.tolist())
        action_turn = T.squeeze(dist_turn).tolist()
        # print(action_turn)
        max = np.amax([abs(a) for a in action_turn])
        # print(max)
        # action_turn = T.squeeze(dist_turn)
        # print(action_turn)
        probs_turn = T.squeeze(T.Tensor( [0.1*action_turn[0] + 0.01*action_turn[1]  + 0.001*action_turn[2]])).item()
        # print(probs_turn)
        # action_turn = T.squeeze(action_turn).item()
        value_turn = T.squeeze(value_turn).item()
        # print("\n", action_turn, "\n")
        return [action_turn,\
                probs_turn,\
                value_turn]

    def learn(self):
        print("learn")
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory_turn.generate_batches()

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
            advantage = T.tensor(advantage).to(self.actor_turn.device)

            values = T.tensor(values).to(self.actor_turn.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor_turn.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor_turn.device)
                actions = T.tensor(action_arr[batch]).to(self.actor_turn.device)

                dist = self.actor_turn(states)
                critic_value = self.critic_turn(states)

                critic_value = T.squeeze(critic_value)
                # print(actions)
                actions = actions.tolist()
                # print(old_probs)
                # print(actions)
                # print(actions[0][2])
                act = []
                for a in actions:
                    act.append(0.1*a[0] + 0.01*a[1] + 0.001*a[2])
                # act = [0.1*actions[:][0] + 0.01*actions[1]  + 0.001*actions[2]]
                new_probs = T.squeeze(T.Tensor(act))
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_turn.optimizer.zero_grad()
                self.critic_turn.optimizer.zero_grad()
                total_loss.backward()
                self.actor_turn.optimizer.step()
                self.critic_turn.optimizer.step()

        self.memory_turn.clear_memory()  
    