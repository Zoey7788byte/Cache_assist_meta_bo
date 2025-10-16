import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import hashlib
from parameters import args_parser
args = args_parser()
import time

# 启用全局性能优化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(2)  # 限制CPU线程数，避免过度并行

class ParallelSampler:
    def __init__(self, network, max_batch_size=64):
        self.network = network
        self.max_batch_size = max_batch_size
        self._cached_noise = {}  # 预生成噪声缓存
        self._noise_pool_size = 1000
        
    def parallel_sample(self, state_tensor, num_samples):
        """极速并行采样 - 预分配所有内存"""
        if num_samples == 1:
            return self._single_sample_optimized(state_tensor)
        
        with torch.no_grad():
            # 一次性批量处理
            batch_state = state_tensor.unsqueeze(0).expand(num_samples, *state_tensor.shape)
            return self.network(batch_state, sample=True)
    
    def _single_sample_optimized(self, state_tensor):
        """单样本优化路径"""
        with torch.no_grad():
            return self.network(state_tensor.unsqueeze(0), sample=True).squeeze(0)

class UltraFastVectorizedUpdater:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        # 预计算常用数组索引
        self._precompute_indices()
        
    def _precompute_indices(self):
        """预计算索引以避免运行时计算"""
        if hasattr(self.env, 'NUM_EDGE_NODES'):
            self.N = self.env.NUM_EDGE_NODES
            self.M = self.env.NUM_MODELS
            # 预计算常用张量形状
            self._deployment_shape = (self.N, self.M)
    
    def lightning_state_update(self, hard_deployment, hard_update, time_step):
        """闪电般的状态更新 - 最小化数据转换"""
        try:
            # 直接使用GPU张量进行更新，避免CPU转换
            if hard_deployment.is_cuda:
                hard_deployment_cpu = hard_deployment.detach().cpu().numpy().astype(np.int32)
            else:
                hard_deployment_cpu = hard_deployment.numpy().astype(np.int32)
            
            if hard_update.is_cuda:
                hard_update_cpu = hard_update.detach().cpu().numpy().astype(np.float32)
            else:
                hard_update_cpu = hard_update.numpy().astype(np.float32)
            
            # 备份并更新状态（最小化操作）
            backup = self.env.cache_state.copy() if hasattr(self.env, 'cache_state') else None
            
            # 批量更新
            if hasattr(self.env, 'cache_state'):
                self.env.cache_state = hard_deployment_cpu
            
            # 快速精度更新
            if hasattr(self.env, 'update_accuracies'):
                self.env.update_accuracies(time_step, hard_update_cpu, backup)
            
            # 批量获取状态（避免重复转换）
            age_state = torch.from_numpy(self.env.age_state).to(self.device, dtype=torch.float32, non_blocking=True)
            accuracies = torch.from_numpy(self.env.accuracies).to(self.device, dtype=torch.float32, non_blocking=True)
            
            return age_state, accuracies, backup
            
        except Exception as e:
            return None, None, None
    
    def restore_state(self, backup):
        """快速状态恢复"""
        if backup is not None and hasattr(self.env, 'cache_state'):
            self.env.cache_state = backup

class HyperOptimizedImportanceWeightedSampler:
    def __init__(self, network, num_samples=5, temperature=1.0):  # 减少默认样本数
        self.network = network
        self.num_samples = num_samples
        self.temperature = temperature
        self.sampler = ParallelSampler(network)
        # 预分配计算张量
        self._weight_cache = {}
        
    def sample_with_importance_weights(self, state_tensor):
        """超优化的重要性加权采样"""
        # 获取样本
        samples = self.sampler.parallel_sample(state_tensor, self.num_samples)
        
        # 确保正确形状，使用inplace操作
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)
        elif samples.dim() > 2:
            samples = samples.view(samples.size(0), -1)
        
        # 极速对数概率计算
        log_probs = self._ultra_fast_log_probs(samples)
        
        # 数值稳定的softmax（inplace操作）
        weights = F.softmax(log_probs / self.temperature, dim=0)
        
        return samples, weights
    
    def _ultra_fast_log_probs(self, samples):
        """超快对数概率计算 - 避免复杂运算"""
        # 简化的高斯假设，使用L2范数
        return -0.5 * torch.norm(samples, dim=1, p=2) / self.temperature

class LightweightCache:
    def __init__(self, max_size=30):  # 减小缓存大小
        self.cache = {}
        self.access_count = {}
        self.max_size = max_size
        
    def get(self, key):
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_size:
            # 移除访问次数最少的项
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value.clone() if hasattr(value, 'clone') else value
        self.access_count[key] = 1

class UltraFastSamplingStrategy:
    def __init__(self, network):
        self.network = network
        self.cache = LightweightCache(max_size=20)  # 更小的缓存
        self.sampler = ParallelSampler(network)
        
    def get_sample_fast(self, state_tensor, num_samples=1):
        """极速采样获取"""
        # 简化的哈希键
        state_hash = hash((state_tensor.sum().item(), state_tensor.std().item(), state_tensor.shape[0]))
        
        cached = self.cache.get(state_hash)
        if cached is not None:
            return cached[:num_samples] if cached.size(0) >= num_samples else cached
        
        # 新采样
        samples = self.sampler.parallel_sample(state_tensor, num_samples)
        
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)
        
        self.cache.put(state_hash, samples)
        return samples

class BayesianMetaPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, prior_mean=None, 
                 prior_std=0.1, initial_logvar=-3.0, init_strategy='meta_learning',
                 init_seed=42): 
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.prior_std = prior_std
        self.initial_logvar = initial_logvar
        self.init_strategy = init_strategy
        self.init_seed = init_seed 
        
        # 定义变分参数
        self.fc1_mean = nn.Linear(input_dim, self.hidden_dim)
        self.fc1_logvar = nn.Linear(input_dim, self.hidden_dim)
        
        self.fc2_mean = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2_logvar = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.fc_out_mean = nn.Linear(self.hidden_dim, output_dim)
        self.fc_out_logvar = nn.Linear(self.hidden_dim, output_dim)
        
        self.logit_scale = nn.Parameter(torch.tensor(0.1))
        self.variance_adjustment = nn.Parameter(torch.tensor(1.0))
        
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        
        self._initialize_with_isolated_seed(prior_mean, prior_std)

    def _get_deterministic_seed(self, strategy_name):
        """根据策略名称生成确定性种子"""
        hash_obj = hashlib.md5(f"{strategy_name}_{self.init_seed}".encode())
        return int(hash_obj.hexdigest()[:8], 16)

    def _initialize_with_isolated_seed(self, prior_mean, prior_std):
        """用隔离的随机数生成器进行初始化"""
        # 保存当前全局随机状态
        cpu_state = torch.get_rng_state()
        cuda_states = []
        if torch.cuda.is_available():
            cuda_states = [torch.cuda.get_rng_state(i) 
                          for i in range(torch.cuda.device_count())]
        
        try:
            # 生成策略特定的种子
            strategy_seed = self._get_deterministic_seed(self.init_strategy)
            
            # 设置临时种子
            torch.manual_seed(strategy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(strategy_seed)
            
            # 执行初始化
            if self.init_strategy == 'meta_learning':
                self._initialize_parameters()
                if prior_mean is not None:
                    self.init_with_prior(prior_mean)
                    print(f"[初始化] 已加载元学习先验参数 (种子: {strategy_seed})")
                else:
                    print(f"[初始化] 警告: 未提供元学习先验，使用Xavier初始化 (种子: {strategy_seed})")
                    
            elif self.init_strategy == 'random':
                self._initialize_random()
                print(f"[初始化] 使用随机初始化 (种子: {strategy_seed})")
                
            elif self.init_strategy == 'zero':
                self._initialize_zero()
                print(f"[初始化] 使用小随机值初始化 (种子: {strategy_seed})")
                
            elif self.init_strategy == 'standard_prior':
                self._initialize_standard_prior()
                print(f"[初始化] 使用标准正态先验初始化 (种子: {strategy_seed})")
            else:
                print(f"[警告] 未知策略 '{self.init_strategy}'，使用默认Xavier初始化")
                self._initialize_parameters()
        
        finally:
            torch.set_rng_state(cpu_state)
            if torch.cuda.is_available():
                for i, state in enumerate(cuda_states):
                    torch.cuda.set_rng_state(state, i)
            
    def _initialize_parameters(self):
        """Xavier初始化（元学习的基础）"""
        nn.init.xavier_uniform_(self.fc1_mean.weight)
        nn.init.zeros_(self.fc1_mean.bias)
        nn.init.xavier_uniform_(self.fc2_mean.weight)
        nn.init.zeros_(self.fc2_mean.bias)
        nn.init.xavier_uniform_(self.fc_out_mean.weight, gain=0.01)
        nn.init.zeros_(self.fc_out_mean.bias)
        
        nn.init.constant_(self.fc1_logvar.weight, self.initial_logvar)
        nn.init.constant_(self.fc1_logvar.bias, self.initial_logvar)
        nn.init.constant_(self.fc2_logvar.weight, self.initial_logvar)
        nn.init.constant_(self.fc2_logvar.bias, self.initial_logvar)
        nn.init.constant_(self.fc_out_logvar.weight, self.initial_logvar)
        nn.init.constant_(self.fc_out_logvar.bias, self.initial_logvar)
    
    def _initialize_random(self):
        """大幅度随机初始化"""
        nn.init.kaiming_normal_(self.fc1_mean.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.fc1_mean.bias, -1.0, 1.0)
        nn.init.kaiming_normal_(self.fc2_mean.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.fc2_mean.bias, -1.0, 1.0)
        nn.init.kaiming_normal_(self.fc_out_mean.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.fc_out_mean.bias, -2.0, 2.0)
        
        # 大范围的对数方差
        nn.init.uniform_(self.fc1_logvar.weight, -1.5, -0.5)
        nn.init.uniform_(self.fc1_logvar.bias, -1.5, -0.5)
        nn.init.uniform_(self.fc2_logvar.weight, -1.5, -0.5)
        nn.init.uniform_(self.fc2_logvar.bias, -1.5, -0.5)
        nn.init.uniform_(self.fc_out_logvar.weight, -1.5, -0.5)
        nn.init.uniform_(self.fc_out_logvar.bias, -1.5, -0.5)

    def _initialize_zero(self):
        """接近零的小随机初始化"""
        nn.init.normal_(self.fc1_mean.weight, mean=0.0, std=0.005)
        nn.init.normal_(self.fc1_mean.bias, mean=0.0, std=0.001)
        nn.init.normal_(self.fc2_mean.weight, mean=0.0, std=0.005)
        nn.init.normal_(self.fc2_mean.bias, mean=0.0, std=0.001)
        nn.init.normal_(self.fc_out_mean.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.fc_out_mean.bias, mean=0.0, std=0.005)
        
        # 高不确定性（log(1)=0 表示std=1）
        nn.init.constant_(self.fc1_logvar.weight, 0.0)
        nn.init.constant_(self.fc1_logvar.bias, 0.0)
        nn.init.constant_(self.fc2_logvar.weight, 0.0)
        nn.init.constant_(self.fc2_logvar.bias, 0.0)
        nn.init.constant_(self.fc_out_logvar.weight, 0.0)
        nn.init.constant_(self.fc_out_logvar.bias, 0.0)

    def _initialize_standard_prior(self):
        """标准正态先验初始化"""
        nn.init.normal_(self.fc1_mean.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.fc1_mean.bias, mean=0.0, std=0.2)
        nn.init.normal_(self.fc2_mean.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.fc2_mean.bias, mean=0.0, std=0.2)
        nn.init.normal_(self.fc_out_mean.weight, mean=0.0, std=1.5)
        nn.init.normal_(self.fc_out_mean.bias, mean=0.0, std=0.5)
        
        # 标准差为1 (log(1)=0)
        nn.init.constant_(self.fc1_logvar.weight, 0.0)
        nn.init.constant_(self.fc1_logvar.bias, 0.0)
        nn.init.constant_(self.fc2_logvar.weight, 0.0)
        nn.init.constant_(self.fc2_logvar.bias, 0.0)
        nn.init.constant_(self.fc_out_logvar.weight, 0.0)
        nn.init.constant_(self.fc_out_logvar.bias, 0.0)
    
    def init_with_prior(self, prior_mean):
        """使用元学习参数初始化"""
        with torch.no_grad():
            param_mapping = {
                'fc1_mean.weight': self.fc1_mean.weight,
                'fc1_mean.bias': self.fc1_mean.bias,
                'fc2_mean.weight': self.fc2_mean.weight,
                'fc2_mean.bias': self.fc2_mean.bias,
                'fc_out_mean.weight': self.fc_out_mean.weight,
                'fc_out_mean.bias': self.fc_out_mean.bias,
            }
            
            for prior_key, target_param in param_mapping.items():
                if prior_key in prior_mean:
                    try:
                        source_param = prior_mean[prior_key]
                        if source_param.shape == target_param.shape:
                            target_param.copy_(source_param)
                    except Exception as e:
                        print(f"参数初始化警告 {prior_key}: {e}")
      
    def kl_divergence(self, prior_mean=0.0, prior_std=1.0):
        """修正：正确的KL散度计算"""
        kl_total = 0.0
        
        # 计算所有权重和偏置的KL散度
        for mean_param, logvar_param in [
            (self.fc1_mean.weight, self.fc1_logvar.weight),
            (self.fc1_mean.bias, self.fc1_logvar.bias),
            (self.fc2_mean.weight, self.fc2_logvar.weight),
            (self.fc2_mean.bias, self.fc2_logvar.bias),
            (self.fc_out_mean.weight, self.fc_out_logvar.weight),
            (self.fc_out_mean.bias, self.fc_out_logvar.bias)
        ]:
            posterior_mean = mean_param
            posterior_logvar = logvar_param
            posterior_var = torch.exp(posterior_logvar)
            
            # 与先验分布的KL散度
            kl = -0.5 * torch.sum(1 + posterior_logvar - 
                                (posterior_mean - prior_mean)**2 / (prior_std**2) - 
                                posterior_var / (prior_std**2))
            kl_total += kl
        
        return kl_total
    
    def sample_parameter_improved(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar) * self.variance_adjustment
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def init_with_prior(self, prior_mean):
        """使用元学习参数初始化"""
        with torch.no_grad():
            param_mapping = {
                'fc1_mean.weight': self.fc1_mean.weight,
                'fc1_mean.bias': self.fc1_mean.bias,
                'fc2_mean.weight': self.fc2_mean.weight,
                'fc2_mean.bias': self.fc2_mean.bias,
                'fc_out_mean.weight': self.fc_out_mean.weight,
                'fc_out_mean.bias': self.fc_out_mean.bias,
            }
            
            for prior_key, target_param in param_mapping.items():
                if prior_key in prior_mean:
                    try:
                        source_param = prior_mean[prior_key]
                        if source_param.shape == target_param.shape:
                            target_param.copy_(source_param)
                        else:
                            # 尝试reshape或部分复制
                            if source_param.numel() == target_param.numel():
                                target_param.data.copy_(source_param.reshape(target_param.shape))
                    except Exception as e:
                        print(f"参数初始化警告 {prior_key}: {e}")
    
    def forward(self, x, sample=True):
        # 第一层
        if sample and self.training:
            fc1_weight = self.sample_parameter_improved(self.fc1_mean.weight, self.fc1_logvar.weight)
            fc1_bias = self.sample_parameter_improved(self.fc1_mean.bias, self.fc1_logvar.bias)
        else:
            fc1_weight = self.fc1_mean.weight
            fc1_bias = self.fc1_mean.bias
        
        x = F.linear(x, fc1_weight, fc1_bias)
        x = F.relu(self.ln1(x))
        
        # 第二层
        if sample and self.training:
            fc2_weight = self.sample_parameter_improved(self.fc2_mean.weight, self.fc2_logvar.weight)
            fc2_bias = self.sample_parameter_improved(self.fc2_mean.bias, self.fc2_logvar.bias)
        else:
            fc2_weight = self.fc2_mean.weight
            fc2_bias = self.fc2_mean.bias
        
        x = F.linear(x, fc2_weight, fc2_bias)
        x = F.relu(self.ln2(x))
        
        # 输出层
        if sample and self.training:
            fc_out_weight = self.sample_parameter_improved(self.fc_out_mean.weight, self.fc_out_logvar.weight)
            fc_out_bias = self.sample_parameter_improved(self.fc_out_mean.bias, self.fc_out_logvar.bias)
        else:
            fc_out_weight = self.fc_out_mean.weight
            fc_out_bias = self.fc_out_mean.bias
        
        output = F.linear(x, fc_out_weight, fc_out_bias) * torch.sigmoid(self.logit_scale)
        return output

class MetaLearningBayesianOptimizer:
    def __init__(self, env, args, meta_learner, alg):
        self.env = env
        self.args = args
        self.meta_learner = meta_learner
        self.N = env.NUM_EDGE_NODES
        self.M = env.NUM_MODELS
        # 修改：当meta_learner为None时，使用env的device或默认device
        if meta_learner is not None:
            self.device = meta_learner.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.device = meta_learner.device
        self.alg = alg  # 保存算法类型
        
        # 启用所有可能的性能优化
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
        
        # 根据alg参数确定初始化策略和先验参数
        init_strategy, meta_params, prior_std = self._determine_init_strategy(alg)
        
        input_dim = self._calculate_input_dim()
        output_dim = self.N * self.M * 2
        
        # 创建贝叶斯策略网络，传入初始化策略
        self.bayesian_policy = BayesianMetaPolicyNetwork(
            input_dim, output_dim, hidden_dim=args.HIDDEN_DIM,
            prior_mean=meta_params,
            prior_std=prior_std,
            initial_logvar=getattr(args, 'initial_logvar', -3.0),
            init_strategy=init_strategy
        ).to(self.device)
        
        # 优化器设置
        vi_lr = getattr(args, 'vi_lr', 0.001)
        self.optimizer = torch.optim.AdamW(
            self.bayesian_policy.parameters(),
            lr=vi_lr * 5,
            weight_decay=1e-6,
            amsgrad=True
        )
        
        # MC采样设置
        num_mc_samples = getattr(args, 'num_mc_samples', 10)
        self.num_mc_samples = max(1, num_mc_samples // 10)
        
        # 初始化高性能组件
        self.sampling_strategy = UltraFastSamplingStrategy(self.bayesian_policy)
        self.importance_sampler = HyperOptimizedImportanceWeightedSampler(
            self.bayesian_policy, num_samples=self.num_mc_samples
        )
        self.vectorized_updater = UltraFastVectorizedUpdater(env, self.device)
        
        # 训练参数
        self.kl_weight = getattr(args, 'initial_kl_weight', 0.1)
        self.training_step = 0
        
        # 预分配常用张量
        self._preallocate_tensors()
        
        # 预计算静态数据
        self._precompute_static_data()
        
        # 性能监控
        self.total_time = 0
        self.call_count = 0
        self.skip_update = False
    
    def _determine_init_strategy(self, alg):
        if alg == 'MEBO':
            init_strategy = 'meta_learning'
            meta_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
            prior_std = getattr(self.args, 'prior_std', 0.1)
            print(f"[{alg}] 使用元学习先验初始化")
            
        elif alg == 'random_BO':
            init_strategy = 'random'
            meta_params = None
            prior_std = getattr(self.args, 'prior_std', 0.1)
            print(f"[{alg}] 使用随机初始化")
            
        elif alg == 'zero_BO':
            init_strategy = 'zero'
            meta_params = None
            prior_std = getattr(self.args, 'prior_std', 0.1)
            print(f"[{alg}] 使用零初始化")
            
        elif alg == 'standard_prior_BO':
            init_strategy = 'standard_prior'
            meta_params = None
            prior_std = 1.0
            print(f"[{alg}] 使用标准正态先验初始化 (mean=0, std=1)")
            
        else:
            print(f"[警告] 未知算法类型 '{alg}'，使用默认元学习初始化")
            init_strategy = 'meta_learning'
            meta_params = copy.deepcopy(self.meta_learner.meta_policy.state_dict())
            prior_std = getattr(self.args, 'prior_std', 0.1)
        
        return init_strategy, meta_params, prior_std
    
    def _preallocate_tensors(self):
        """预分配常用张量以减少内存分配开销"""
        self.zeros_deployment = torch.zeros((self.N, self.M), dtype=torch.int32, device=self.device)
        self.zeros_update = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.device)
        self.temp_tensor = torch.empty((self.N * self.M * 2,), dtype=torch.float32, device=self.device)
    
    def _precompute_static_data(self):
        """预计算静态数据"""
        try:
            if hasattr(self.env, 'cloud_manager') and hasattr(self.env.cloud_manager, 'model_sizes'):
                self.model_sizes = torch.tensor(
                    self.env.cloud_manager.model_sizes, 
                    dtype=torch.float32, device=self.device
                )
            else:
                self.model_sizes = torch.ones(self.M, dtype=torch.float32, device=self.device)
            
            if hasattr(self.env, 'get_switch_cost'):
                self.switch_costs = torch.tensor([
                    self.env.get_switch_cost(i) for i in range(self.M)
                ], dtype=torch.float32, device=self.device)
            else:
                self.switch_costs = torch.ones(self.M, dtype=torch.float32, device=self.device)
        except Exception as e:
            print(f"警告：预计算静态数据时出现问题: {e}")
            self.model_sizes = torch.ones(self.M, dtype=torch.float32, device=self.device)
            self.switch_costs = torch.ones(self.M, dtype=torch.float32, device=self.device)
    
    def _calculate_input_dim(self):
        if hasattr(self.env, 'get_state_vector'):
            return len(self.env.get_state_vector(0))
        return self.N * self.M * 2
    
    def _to_tensor_fast(self, data):
        """超快张量转换"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device, non_blocking=True)
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)
    
    def _sample_and_constrain_ultra_fast(self, state_tensor):
        # 直接采样
        output = self.bayesian_policy(state_tensor.unsqueeze(0), sample=True).squeeze(0)
        
        mid_point = self.N * self.M
        
        # 确保输出长度正确
        if output.numel() < mid_point * 2:
            padding = torch.zeros(mid_point * 2 - output.numel(), device=output.device)
            output = torch.cat([output, padding])
        
        deploy_logits = output[:mid_point].view(self.N, self.M)
        update_logits = output[mid_point:].view(self.N, self.M)
        
        # 快速约束应用
        deployed_surrogate, hard_deployment = self.meta_learner.enforce_storage_constraints(deploy_logits)
        update_surrogate, hard_update = self.meta_learner.enforce_update_constraints(
            update_logits, deployed_surrogate
        )
        
        return deployed_surrogate, hard_deployment, update_surrogate, hard_update
    
    def compute_cost_lightning_fast_tensor(self, state_tensor, request_tensor, time_step, prev_deployment=None):
        """闪电般的成本计算 - 返回保持梯度的张量"""
        total_cost = torch.tensor(0.0, device=self.device, requires_grad=True)
        valid_samples = 0
        
        # 使用单一样本快速估计
        for _ in range(max(1, self.num_mc_samples)):
            backup = None
            try:
                deployed_surrogate, hard_deployment, update_surrogate, hard_update = \
                    self._sample_and_constrain_ultra_fast(state_tensor)
                
                # 快速状态更新
                updated_age_state, updated_accuracies, backup = \
                    self.vectorized_updater.lightning_state_update(hard_deployment, hard_update, time_step)
                
                if updated_accuracies is not None:
                    cost, _, _, _, _, _, _ = self.meta_learner.calculate_cost_with_gradients(
                        deployed_surrogate, update_surrogate, request_tensor, time_step,
                        prev_deployment, updated_age_state, updated_accuracies.view(self.N, self.M)
                    )
                    
                    if torch.isfinite(cost):
                        total_cost = total_cost + cost
                        valid_samples += 1
                        
                        # 早期终止机制
                        if valid_samples >= 1:
                            break
                
            except Exception:
                continue
            finally:
                if backup is not None:
                    self.vectorized_updater.restore_state(backup)
        
        return total_cost / max(1, valid_samples)

    def compute_cost_lightning_fast(self, state_tensor, request_tensor, time_step, prev_deployment=None):
        """闪电般的成本计算 - 兼容旧版本接口"""
        tensor_result = self.compute_cost_lightning_fast_tensor(state_tensor, request_tensor, time_step, prev_deployment)
        return tensor_result.item()
    
    def variational_update_turbo(self, state_tensor, request_tensor, time_step, prev_deployment=None):
        """修正：使用正确的KL散度计算"""
        self.optimizer.zero_grad(set_to_none=True)
        
        # 计算成本
        expected_cost_tensor = self.compute_cost_lightning_fast_tensor(
            state_tensor, request_tensor, time_step, prev_deployment
        )
        
        # 修正：使用正确的KL散度计算
        kl = self.bayesian_policy.kl_divergence()
        
        # 目标函数
        loss = expected_cost_tensor + self.kl_weight * kl
        
        # 检查数值稳定性并进行反向传播
        if torch.isfinite(loss):
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.bayesian_policy.parameters(), 1.0)
            
            self.optimizer.step()
            self.training_step += 1
        
        return loss.item(), expected_cost_tensor.item(), kl.item()
    
    def get_decision_ultra_fast(self, state_tensor, request_tensor, time_step, prev_deployment=None):
        """超快决策获取"""
        with torch.no_grad():
            # 使用确定性输出
            output = self.bayesian_policy(state_tensor.unsqueeze(0), sample=False).squeeze(0)
            
            mid_point = self.N * self.M
            
            if output.numel() >= mid_point * 2:
                deploy_logits = output[:mid_point].view(self.N, self.M)
                update_logits = output[mid_point:].view(self.N, self.M)
            else:
                deploy_logits = torch.zeros(self.N, self.M, device=output.device)
                update_logits = torch.zeros(self.N, self.M, device=output.device)
            
            # 应用约束
            _, hard_deployment = self.meta_learner.enforce_storage_constraints(deploy_logits)
            _, hard_update = self.meta_learner.enforce_update_constraints(update_logits, hard_deployment)
            
            # 转换为numpy
            deployment_np = hard_deployment.detach().cpu().numpy().astype(int)
            update_np = hard_update.detach().cpu().numpy()
            
            return (deployment_np, update_np), 0.0
    
    def optimize(self, t, requests, prev_deployment=None):
        """主优化函数"""
        start_time = time.time()
        
        try:
            # 快速数据准备
            state = self.env.get_state_vector(t)
            state_tensor = self._to_tensor_fast(state)
            request_tensor = self._to_tensor_fast(requests)
            prev_deployment_tensor = self._to_tensor_fast(prev_deployment) if prev_deployment is not None else None
            
            # 获取决策
            (deployment, update_mask), _ = self.get_decision_ultra_fast(
                state_tensor, request_tensor, t, prev_deployment_tensor
            )
            
            # 自适应变分更新
            if not self.skip_update and t % 5 == 0:
                try:
                    loss, cost, kl = self.variational_update_turbo(
                        state_tensor, request_tensor, t, prev_deployment_tensor
                    )
                    
                    if not torch.isfinite(torch.tensor(loss)):
                        self.skip_update = True
                        print(f"警告：第{t}步检测到异常loss，暂停更新")
                except Exception as e:
                    if t % 100 == 0:
                        print(f"更新异常 (步骤{t}): {e}")
            
            # 重新启用更新的条件
            if self.skip_update and t % 50 == 0:
                self.skip_update = False
            
            # 性能监控
            elapsed = time.time() - start_time
            self.total_time += elapsed
            self.call_count += 1
            
            if t % 500 == 0 and self.call_count > 0:
                avg_time = self.total_time / self.call_count
                print(f"步骤 {t}: 平均耗时 {avg_time:.4f}s, 总调用 {self.call_count}")
                if self.skip_update:
                    print("   (当前跳过变分更新)")
            
            return deployment, update_mask
            
        except Exception as e:
            if t % 100 == 0:
                print(f"优化步骤{t}出错: {e}")
            # 快速回退
            deployment = self.zeros_deployment.cpu().numpy()
            update_mask = self.zeros_update.cpu().numpy()
            return deployment, update_mask
