# 导入必要的库
import numpy as np               # 用于数值计算
import torch                     # PyTorch库，处理深度学习
import torch.nn as nn            # 神经网络模块
import torch.optim as optim      # 优化器
import random                    # 用于随机化
from collections import deque    # 用于存储经验回放池
import matplotlib.pyplot as plt  # 用于绘制结果
from IPython import display      # 引入IPython库中的display模块，用来在Jupyter中清除输出
class LunarEnvironment:
    def __init__(self, grid_size=(10, 10), obstacles=None, goal=(9, 9)):
        """
        初始化环境参数。
        :param grid_size: 网格的大小，默认为10x10。
        :param obstacles: 障碍物的位置列表，默认为None。
        :param goal: 目标点的坐标，默认为(9, 9)。
        """
        self.grid_size = grid_size  # 设置网格的大小
        # 如果没有提供障碍物，则使用默认的障碍物位置
        self.obstacles = obstacles if obstacles else [(3, 3), (3, 4), (4, 4), (5, 4), (6, 4), (6, 3), (6, 2)]
        self.goal = goal  # 设置目标点位置
        self.state = (0, 0)  # 起点位置（0, 0）
        # 定义可选择的动作：上、下、左、右
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        """重置环境，返回起点位置"""
        self.state = (0, 0)  # 将状态重置为起点位置
        return self.state  # 返回当前状态

    def step(self, action):
        """
        根据当前状态和动作执行一步。
        :param action: 动作，'up', 'down', 'left', 'right'之一
        :return: 下一状态，奖励，是否完成任务
        """
        x, y = self.state  # 获取当前状态的坐标
        # 根据动作更新坐标
        if action == 'up':
            x = max(0, x - 1)  # 向上移动，确保不超出边界
        elif action == 'down':
            x = min(self.grid_size[0] - 1, x + 1)  # 向下移动，确保不超出边界
        elif action == 'left':
            y = max(0, y - 1)  # 向左移动，确保不超出边界
        elif action == 'right':
            y = min(self.grid_size[1] - 1, y + 1)  # 向右移动，确保不超出边界

        next_state = (x, y)  # 更新下一状态

        # 奖励函数
        if next_state == self.goal:
            reward = 100  # 到达目标点时给与100的奖励
            done = True    # 任务完成
        elif next_state in self.obstacles:
            reward = -100  # 撞上障碍物时给与-100的惩罚
            done = True    # 任务失败
        else:
            reward = -1  # 每走一步都给-1的惩罚
            done = False  # 任务未完成

        self.state = next_state  # 更新当前状态
        return next_state, reward, done  # 返回下一状态、奖励和任务完成标志

    def render(self):
        """
        可视化环境，显示网格、障碍物、起点和目标点。
        """
        grid = np.zeros(self.grid_size)  # 创建一个全零的网格
        for obs in self.obstacles:
            grid[obs] = -1  # 标记障碍物为-1
        grid[self.goal] = 1  # 标记目标点为1
        grid[self.state] = 0.5  # 标记当前位置为0.5
        plt.imshow(grid, cmap='coolwarm', origin='upper')  # 显示网格

        # 显示图像
        plt.show()

# 创建一个LunarEnvironment环境实例
env = LunarEnvironment()
# 调用render方法可视化环境
env.render()
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化DQN模型。
        :param input_dim: 输入维度，即状态的维度
        :param output_dim: 输出维度，即动作的数量
        """
        super(DQN, self).__init__()
        
        # 神经网络结构：两层全连接层
        self.fc1 = nn.Linear(input_dim, 128)  # 第一层全连接层，输入维度为状态维度，输出128个神经元
        self.fc2 = nn.Linear(128, 128)        # 第二层全连接层，输入128个神经元，输出128个神经元
        self.fc3 = nn.Linear(128, output_dim) # 最后一层全连接层，输出Q值（每个动作的Q值）

    def forward(self, x):
        """
        前向传播函数，定义网络如何处理输入数据
        :param x: 输入的状态
        :return: 每个动作的Q值
        """
        x = torch.relu(self.fc1(x))  # 第一层使用ReLU激活函数
        x = torch.relu(self.fc2(x))  # 第二层使用ReLU激活函数
        x = self.fc3(x)              # 输出层
        return x
class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化经验回放池。
        :param capacity: 回放池的容量
        """
        self.buffer = deque(maxlen=capacity)  # 使用双端队列存储经验
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """
        将一个经验添加到回放池中。
        :param state: 当前状态
        :param action: 执行的动作
        :param reward: 得到的奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从回放池中随机抽取一个批次的经验。
        :param batch_size: 批量大小
        :return: 随机选中的批量经验
        """
        return random.sample(self.buffer, batch_size)

    def size(self):
        """返回回放池当前的经验数量"""
        return len(self.buffer)
def train_dqn(env, model, optimizer, replay_buffer, batch_size=32, gamma=0.99):
    """
    使用DQN算法训练模型。
    :param env: 环境
    :param model: DQN模型
    :param optimizer: 优化器
    :param replay_buffer: 经验回放池
    :param batch_size: 每次训练时的批量大小
    :param gamma: 折扣因子，决定未来奖励的权重
    """
    if replay_buffer.size() < batch_size:
        return  # 如果回放池中的经验不足，不进行训练
    
    # 从回放池中随机选取一个批次的经验
    batch = replay_buffer.sample(batch_size)
    
    # 分别从批次中获取状态、动作、奖励、下一个状态和完成标志
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # 转换成PyTorch的张量
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # 计算当前状态的Q值
    q_values = model(states)
    
    # 获取每个状态对应的动作Q值
    actions = torch.tensor(actions, dtype=torch.long)
    q_values = q_values.gather(1, actions.unsqueeze(1))  # 获取当前选择动作的Q值
    
    # 计算下一个状态的最大Q值（目标Q值）
    next_q_values = model(next_states)
    next_q_values = next_q_values.max(1)[0]  # 取每个下一个状态的最大Q值
    
    # 计算目标Q值
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    
    # 计算损失函数
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    
    # 反向传播，更新网络参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
# 初始化环境
env = LunarEnvironment()

# 初始化DQN模型
input_dim = 2  # 状态是一个二维坐标 (x, y)
output_dim = 4 # 动作数量：上、下、左、右
model = DQN(input_dim, output_dim)

# 初始化优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化经验回放池
replay_buffer = ReplayBuffer(capacity=10000)

# 设置训练参数
batch_size = 32  # 每次训练的批量大小
gamma = 0.99      # 折扣因子，决定未来奖励的权重
epsilon = 0.1     # 探索率，控制智能体的随机行为
epsilon_decay = 0.995  # epsilon 衰减率，逐渐减少探索
min_epsilon = 0.01     # epsilon 最小值
num_episodes = 1000     # 训练的总轮数
def train(env, model, optimizer, replay_buffer, num_episodes, batch_size, gamma, epsilon, epsilon_decay, min_epsilon):
    total_rewards = []  # 存储每个回合的总奖励
    trajectory = []  # 记录轨迹
    
    for episode in range(num_episodes):
        state = env.reset()  # 重置环境，获取初始状态
        episode_reward = 0  # 当前回合的总奖励
        done = False  # 标记任务是否完成
        episode_trajectory = []  # 记录当前回合的轨迹
        
        while not done:
            # epsilon-greedy策略选择动作
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3])  # 随机选择动作
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = model(state_tensor)  # 获取Q值
                    action = q_values.argmax().item()  # 选择Q值最大的动作
            
            # 执行动作，获取下一个状态和奖励
            next_state, reward, done = env.step(env.actions[action])
            
            # 记录当前位置
            episode_trajectory.append(next_state)
            
            # 将经验存入回放池
            replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新当前状态
            state = next_state
            
            # 累加奖励
            episode_reward += reward
            
            # 训练模型
            loss = train_dqn(env, model, optimizer, replay_buffer, batch_size, gamma)
        
        # 在每个回合结束时记录总奖励
        total_rewards.append(episode_reward)
        
        # 保存回合轨迹
        trajectory.append(episode_trajectory)
        
        # 随着训练的进行，逐渐减少epsilon，增加智能体的确定性
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # 每100回合打印一次训练信息
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward}")
    
    return total_rewards, trajectory
# 训练模型
# 调用 train 函数进行模型训练，并将返回的总奖励和轨迹保存到变量中
total_rewards, trajectory = train(
    env,                # 环境对象，模型将在此环境中进行训练
    model,              # 要训练的模型
    optimizer,          # 优化器，用于更新模型参数
    replay_buffer,      # 经验回放缓冲区，存储过去的经验以便后续训练
    num_episodes,       # 训练的总回合数
    batch_size,         # 每次训练时使用的样本数量
    gamma,              # 折扣因子，决定未来奖励的影响程度
    epsilon,            # 探索率，决定模型在训练时选择随机行动的概率
    epsilon_decay,      # 探索率衰减，控制探索率随时间的减少速度
    min_epsilon         # 最小探索率，确保探索率不会下降到这个值以下
)

# 绘制训练过程中的奖励曲线
# 使用 matplotlib 库绘制奖励曲线，显示训练过程中获得的总奖励
plt.plot(total_rewards)  # 绘制总奖励的曲线图

# 设置 x 轴的标签为 'Episode'，表示训练的回合数
plt.xlabel('Episode')

# 设置 y 轴的标签为 'Total Reward'，表示每个回合获得的总奖励
plt.ylabel('Total Reward')

# 设置图表的标题为 'Training Progress'
plt.title('Training Progress')

# 显示绘制的图表
plt.show()
def draw(env, trajectory):
    # 轨迹是一个列表，每个元素代表智能体在某一时刻的位置，环境是一个表示网格的对象
    for i in range(len(trajectory)):  # 遍历轨迹的每一帧（每个时间点的智能体位置）
        plt.cla()  # 清除当前画布中的图像，准备绘制新的一帧
        
        # 创建一个大小为环境网格的数组，用来表示当前的环境
        grid = np.zeros(env.grid_size)  # 初始化一个全是0的数组，表示空地
        # 遍历所有障碍物，并在网格中标记出障碍物的位置
        for obs in env.obstacles:  # `env.obstacles` 存储了障碍物的位置
            grid[obs] = -1  # 将障碍物位置标记为-1
        grid[env.goal] = 1  # 将目标位置标记为1，这里`env.goal`表示目标的位置
        grid[trajectory[i]] = 0.5  # 在当前帧中，将智能体的位置标记为0.5，这里`trajectory[i]`表示智能体的位置

        # 使用imshow函数将网格数据（即环境信息）绘制成图像
        plt.imshow(grid, cmap='coolwarm', origin='upper')  # `cmap='coolwarm'`表示图像颜色映射方式，'upper'表示原点在图像的上方
        plt.title('agent trajectory')  # 设置图像的标题为"agent trajectory"
        display.clear_output(wait=True)  # 清除上一帧的输出，`wait=True`表示等待下一帧输出时再清除
        plt.pause(0.05)  # 暂停0.05秒，控制帧之间的时间间隔，这样每帧可以有足够的时间显示
        
    plt.show()  # 在绘制完所有帧后，显示最终的图像

# 调用该函数，传入环境对象`env`和最后一次训练的轨迹
draw(env, trajectory[-1])
