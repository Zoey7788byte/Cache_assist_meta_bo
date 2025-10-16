import argparse

def args_parser():
    parser = argparse.ArgumentParser() #The content file size

    #System parameter
    parser.add_argument('--time_slot', type=int, default=1000000,
                        help='The model retraing window')
    parser.add_argument('--edge_cache_storage', type=int, default=800,
                        help='The storage of server is 64')
    parser.add_argument('--edge_num', type=int, default=1,
                        help='The num of edge node')

    parser.add_argument('--train_task_num', type=int, default=100,
                        help='The number of tasks at each epoch')
    parser.add_argument('--task_num', type=int, default=20,
                        help='The number task of online test ')
    
    parser.add_argument('--Update_flag', type=bool, default=False,
                        help='The pre test experiment')
    parser.add_argument('--Shared_flag', type=bool, default=False,
                        help='The pre test experiment')


    # 新增：初始化策略参数
    parser.add_argument('--init_strategy', type=str, default='meta_learning',
                       choices=['meta_learning', 'random', 'zero', 'standard_prior', 'custom_prior'],
                       help='网络初始化策略')

    # 新增：自定义先验标准差（当init_strategy='custom_prior'时使用）
    parser.add_argument('--custom_prior_std', type=float, default=0.1,
                       help='自定义先验标准差')
    
    #Meta Parameter
    parser.add_argument('--HIDDEN_DIM', type=int, default=256,
                        help='The network hidden dim')
    parser.add_argument('--num_epochs', type=int, default=130,
                        help='The training epoch')
    parser.add_argument('--task_length', type=int, default=100,
                        help='A batch of task')
    parser.add_argument('--meta_lr', type=float, default=1e-2,
                        help='The learning rate of meta learning')
    parser.add_argument('--inner_lr', type=float, default=1e-1,
                        help='Inner loop learning rate')
    parser.add_argument('--switch_lambda', type=float, default=0.1,
                        help='Inner loop learning rate')
    
    parser.add_argument('--age_decay_factor', type=float, default=0.1,
                        help='Inner loop learning rate')
    parser.add_argument('--max_age', type=int, default=100,
                        help='Inner loop learning rate')

    #Online Meta
    parser.add_argument('--adaptation_steps', type=int, default=10,
                        help='Online learning adapt steps')
    parser.add_argument('--adaptation_slots', type=int, default=5,
                        help='Online learning adapt slots')                    
    parser.add_argument('--adaptation_lr', type=float, default=1e-2,
                        help='Online learning adapt learning rate')
    parser.add_argument('--num_mc_samples', type=float, default=5,
                        help='The Update param')
    parser.add_argument('--cost_temperature', type=float, default=1e-5,
                        help='The Update param')
    parser.add_argument('--prior_std', type=float, default=0.5,
                        help='The Update param')
    parser.add_argument('--vi_lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--initial_logvar', type=float, default=-2.0,
                        help='The Update param')
    parser.add_argument('--sample_cache_size', type=float, default=50,
                        help='The Update param')
    parser.add_argument('--initial_kl_weight', type=float, default=1,
                        help='The Update param')
    parser.add_argument('--kl_weight_decay', type=float, default=0.9995,
                        help='The Update param')
    parser.add_argument('--min_kl_weight', type=float, default=0.01,
                        help='The Update param')
    parser.add_argument('--log_var', type=float, default=0.01,
                        help='The Update param')
                        

    #Online Bo
    parser.add_argument('--prior_variance_scale', type=float, default=0.2,
                        help='Online learning adapt learning rate')
    parser.add_argument('--BO_HIDDEN_DIM', type=int, default=128,
                        help='The network hidden dim')

    #惩罚参数
    parser.add_argument('--lambda_acc', type=float, default=0.2,
                        help='The learning rate of meta learning')     

    parser.add_argument('--INFERENCE_COST_PER_MB', type=float, default=0.002,
                        help='The inference cost per MB')
    parser.add_argument('--COMMUNICATION_COST_PER_MB', type=float, default=0.02,
                        help='The communication cost per MB')
    parser.add_argument('--ACCURACY_COST_PER_MB', type=float, default=0.05,
                        help='The accuracy cost per MB')
    parser.add_argument('--SWITCHING_COST_PER_MB', type=float, default=0.02,
                        help='The switch cost per MB')
    parser.add_argument('--UPDATE_COST_PER_MB', type=float, default=0.01,
                        help='The update cost per MB')
    parser.add_argument('--DELAY_LOCAL', type=float, default=0.001,
                        help='The transfer cost from cloud to edge')
    parser.add_argument('--DELAY_CLOUD', type=float, default=0.1,
                        help='The transfer cost from cloud to edge')
    
     
    #BayesianOptimizer Parameter
    parser.add_argument('--init_points', type=int, default=10,
                        help='The Parameter of BayesianOptimizer')

    #Request Parameter
    parser.add_argument('--base_alpha', type=float, default=2.5,
                        help='The lambda per slot of request')
    parser.add_argument('--expend', type=float, default=1.25,
                        help='The lambda per slot of request')

    #Test Alg Par
    parser.add_argument('--beta', type=float, default=1,
                        help='The lambda per slot of request') 
    parser.add_argument('--rho', type=float, default=0.1,
                        help='The lambda per slot of request')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='The lambda per slot of request')  
    args = parser.parse_args()
    return args