import sys
sys.path.append('/home/leeqh/tianshou')
import os
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.discrete import DQN
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from atari_wrapper import wrap_deepmind
import embedding_prediction
# import tqdm
import pickle
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps_test', type=float, default=0.005)
    parser.add_argument('--eps_train', type=float, default=1.)
    parser.add_argument('--eps_train_final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--target_update_freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step_per_epoch', type=int, default=10000)
    parser.add_argument('--collect_per_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--training_num', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--embedding_path', type=bool, default=False)
    parser.add_argument('--embedding_data_path', type=bool, default=False)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    return parser.parse_args()


def make_atari_env(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack)


def make_atari_env_watch(args):
    return wrap_deepmind(args.task, frame_stack=args.frames_stack,
                         episode_life=False, clip_rewards=False)


def test_dqn(args=get_args()):
    env = make_atari_env(args)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.env.action_space.shape or env.env.action_space.n
    
    # should be N_FRAMES x H x W
    
    print("Observations shape: ", args.state_shape)
    print("Actions shape: ", args.action_shape)
    # make environments
    train_envs = SubprocVectorEnv([lambda: make_atari_env(args)
                                   for _ in range(args.training_num)])
    test_envs = SubprocVectorEnv([lambda: make_atari_env_watch(args)
                                  for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # log
    log_path = os.path.join(args.logdir, args.task, 'embedding')
    embedding_writer = SummaryWriter(log_path + '/without_dropout')

    embedding_net = embedding_prediction.Prediction(*args.state_shape, args.action_shape, args.device).to(device=args.device)

    if args.embedding_path:
        embedding_net.load_state_dict(torch.load(log_path + '/embedding.pth'))
        print("Loaded agent from: ", log_path + '/embedding.pth')
    # numel_list = [p.numel() for p in embedding_net.parameters()]
    # print(sum(numel_list), numel_list)
         
    pre_buffer = ReplayBuffer(args.buffer_size, save_only_last_obs=True, stack_num=args.frames_stack)
    pre_test_buffer = ReplayBuffer(args.buffer_size // 100, save_only_last_obs=True, stack_num=args.frames_stack)
    
    train_collector = Collector(None, train_envs, pre_buffer)
    test_collector = Collector(None, test_envs, pre_test_buffer)    
    if args.embedding_data_path:
        pre_buffer = pickle.load(open(log_path + '/train_data.pkl', 'rb'))
        pre_test_buffer = pickle.load(open(log_path + '/test_data.pkl', 'rb'))
        train_collector.buffer = pre_buffer
        test_collector.buffer = pre_test_buffer
        print('load success')
    else:
        print('collect start')
        train_collector = Collector(None, train_envs, pre_buffer)
        test_collector = Collector(None, test_envs, pre_test_buffer)
        train_collector.collect(n_step=args.buffer_size,random=True)
        test_collector.collect(n_step=args.buffer_size // 100, random=True)
        print(len(train_collector.buffer))
        print(len(test_collector.buffer))
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        pickle.dump(pre_buffer, open(log_path + '/train_data.pkl', 'wb'))
        pickle.dump(pre_test_buffer, open(log_path + '/test_data.pkl', 'wb'))
        print('collect finish')

    #使用得到的数据训练编码网络
    def part_loss(x, device='cpu'):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=device, dtype=torch.float32)
        x = x.view(128, -1)
        temp = torch.cat(((1-x).pow(2.0).unsqueeze_(0),x.pow(2.0).unsqueeze_(0)),dim=0)
        temp_2 = torch.min(temp, dim=0)[0]
        return torch.sum(temp_2)
    
    pre_optim = torch.optim.Adam(embedding_net.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(pre_optim, step_size=50000, gamma=0.1,last_epoch=-1)
    test_batch_data = test_collector.sample(batch_size=0)

    loss_fn = torch.nn.NLLLoss()
    # train_loss = []
    for epoch in range(1, 100001):
        embedding_net.train()
        batch_data = train_collector.sample(batch_size=128)
        # print(batch_data)
        # print(batch_data['obs'][0] == batch_data['obs'][1])
        pred = embedding_net(batch_data['obs'], batch_data['obs_next'])
        x1 = pred[1]
        x2 = pred[2]
        # print(torch.argmax(pred[0], dim=1))
        if not isinstance(batch_data['act'], torch.Tensor):
            act = torch.tensor(batch_data['act'], device=args.device, dtype=torch.int64)
        # print(pred[0].dtype)
        # print(act.dtype)
        # l2_norm = sum(p.pow(2.0).sum() for p in embedding_net.net.parameters())
        # loss = loss_fn(pred[0], act) + 0.001 * (part_loss(x1) + part_loss(x2)) / 64
        loss_1 = loss_fn(pred[0], act)
        loss_2 = 0.01 * (part_loss(x1, args.device) + part_loss(x2, args.device)) / 128
        loss = loss_1 + loss_2
        # print(loss_1)
        # print(loss_2)
        embedding_writer.add_scalar('training loss1', loss_1.item(), epoch)
        embedding_writer.add_scalar('training loss2', loss_2, epoch)
        embedding_writer.add_scalar('training loss', loss.item(), epoch)
        # train_loss.append(loss.detach().item())
        pre_optim.zero_grad()
        loss.backward()
        pre_optim.step()
        scheduler.step()

        if epoch % 10000 == 0 or epoch == 1:
            print(pre_optim.state_dict()['param_groups'][0]['lr'])  
            # print("Epoch: %d,Train: Loss: %f" % (epoch, float(loss.item())))
            correct = 0
            numel_list = [p for p in embedding_net.parameters()][-2]
            print(numel_list)
            embedding_net.eval()
            with torch.no_grad():
                test_pred, x1, x2, _ = embedding_net(test_batch_data['obs'], test_batch_data['obs_next'])
                if not isinstance(test_batch_data['act'], torch.Tensor):
                    act = torch.tensor(test_batch_data['act'], device=args.device, dtype=torch.int64)
                loss_1 = loss_fn(test_pred, act)
                loss_2 = 0.01 * (part_loss(x1, args.device) + part_loss(x2, args.device)) / 128
                loss = loss_1 + loss_2  
                embedding_writer.add_scalar('test loss', loss.item(), epoch)       
                # print("Test Loss: %f" % (float(loss)))                   
                print(torch.argmax(test_pred,dim=1))
                print(act)
                correct += int((torch.argmax(test_pred,dim=1) == act).sum())
                print('Acc:',correct / len(test_batch_data))
    

    torch.save(embedding_net.state_dict(), os.path.join(log_path, 'embedding.pth'))
    embedding_writer.close()  
    # plt.figure()
    # plt.plot(np.arange(100000),train_loss)
    # plt.show()
    exit()
    #构建hash表


    # log
    log_path = os.path.join(args.logdir, args.task, 'dqn')
    writer = SummaryWriter(log_path)

    # define model
    net = DQN(*args.state_shape, args.action_shape, args.device).to(device=args.device)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(net, optim, args.gamma, args.n_step,
                       target_update_freq=args.target_update_freq)
    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path))
        print("Loaded agent from: ", args.resume_path)

    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    pre_buffer.reset()
    buffer = ReplayBuffer(args.buffer_size, ignore_obs_next=True,
                          save_only_last_obs=True, stack_num=args.frames_stack)
    # collector
    # train_collector中传入preprocess_fn对奖励进行重构
    train_collector = Collector(policy, train_envs, buffer)
    test_collector = Collector(policy, test_envs)


    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))
    
    def stop_fn(x):
        if env.env.spec.reward_threshold:
            return x >= env.spec.reward_threshold
        elif 'Pong' in args.task:
            return x >= 20

    def train_fn(x):
        # nature DQN setting, linear decay in the first 1M steps
        now = x * args.collect_per_step * args.step_per_epoch
        if now <= 1e6:
            eps = args.eps_train - now / 1e6 * \
                (args.eps_train - args.eps_train_final)
            policy.set_eps(eps)
        else:
            policy.set_eps(args.eps_train_final)
        print("set eps =", policy.eps)

    def test_fn(x):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Testing agent ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        test_collector.reset()
        result = test_collector.collect(n_episode=[1] * args.test_num,
                                        render=1/30)
        pprint.pprint(result)

    if args.watch:
        watch()
        exit(0)


    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * 4)
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.test_num,
        args.batch_size, train_fn=train_fn, test_fn=test_fn,
        stop_fn=stop_fn, save_fn=save_fn, writer=writer, test_in_train=False)

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    test_dqn(get_args())
