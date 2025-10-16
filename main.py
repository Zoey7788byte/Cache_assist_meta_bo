from CloudEdgeCoordinator import CloudEdgeCoordinator
from cloud_manager import CloudModelManager
import torchvision.models as models
from DatasetManager import RequestGenerator
import torch
from EdgeEnvironment import EdgeEnvironment
import numpy as np
from scipy.stats import entropy
import time
from BayesianOptimizer import BayesianOptimizer
import pandas as pd
from compare_Alg.Greedy import GreedyCacheOptimizer
from compare_Alg.Random import RandomCachingPlacement
from compare_Alg.Meta_OCO import MetaOCO
from compare_Alg.LRU import LRUCacheOptimizer
from compare_Alg.MPUTA import MPUTADeployment
from compare_Alg.Meta import MetaOptimization
from compare_Alg.MeBO import MetaLearningBayesianOptimizer
from check import check_model_parameters
from MAML import MetaLearner
import torch.nn.functional as F
import os
from parameters import args_parser
args = args_parser()

# ================== 使用示例 ==================
def main_simulation():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化云服务
    cloud_manager = CloudModelManager()
    cloud_edge_coordinator = CloudEdgeCoordinator(cloud_manager)

    #注册边缘节点
    for edgeNodeId in range(args.edge_num):
        cloud_edge_coordinator.register_edge_node(node_id= edgeNodeId, max_storage=args.edge_cache_storage) 
    cloud_manager.register_trained_models()

    NUM_MODELS = cloud_manager.num_shared + cloud_manager.num_specific
    print("NUM_MODELS: ",NUM_MODELS, cloud_manager.num_shared, cloud_manager.num_specific)
    print(f"配置: {args.edge_num}节点, {cloud_manager.num_specific}模型, {args.time_slot}时间槽")
    print(f"共享模型: {cloud_manager.num_shared}, 特定模型: {cloud_manager.num_specific}")
    
    # 创建请求生成器
    request_generator = RequestGenerator(
       cloud_manager.num_shared,
        cloud_manager.num_specific,
       cloud_manager.model_db
    )
    all_requests, request_records, history_records, _ = request_generator.generate_requests(args.edge_num, args.time_slot)
    env = EdgeEnvironment(cloud_edge_coordinator, cloud_manager, request_records, args.edge_num, args.Shared_flag)
    
    # 性能记录
    performance_history = []
    start_time = time.time()

    #贝叶斯初始化测试。定义要测试的策略
    strategies_to_test = {
        'meta_learning': {},
        'random': {},
        'zero': {},
        'standard_prior': {},
        'custom_prior_0.01': {'custom_prior_std': 0.01},
        'custom_prior_0.1': {'custom_prior_std': 0.1},
        'custom_prior_0.5': {'custom_prior_std': 0.5},
        'custom_prior_1.0': {'custom_prior_std': 1.0},
    }

    Alg = ['Meta','MPUTA','Random','Greedy','Meta_OCO','LRU','MEBO']
    # Alg = [ "random_BO", "zero_BO", "standard_prior_BO"]
    # Alg = ['Meta_OCO','Random', 'LRU','MEBO','Meta']
    # Alg = ['LRU']
    # Alg = ['Meta_OCO']
    # Alg = ['Greedy']
    # Alg = ['MPUTA']
    # Alg = ['LRU']
    # Alg = ['Meta_OCO']
    # Alg = ['MEBO']
    # Alg = ['Greedy','Random','LRU']
    train_end_time = 0
    train_end_time = int(args.task_length * args.train_task_num)
    
    for alg in Alg:
        iter_start = time.time()
        if alg in ["Meta", "MEBO","random_BO", "zero_BO", "standard_prior_BO"]:
            
            meta_learner = MetaLearner(env, all_requests, request_records, train_end_time, num_epochs=args.num_epochs, task_length=args.task_length) #这里定义了task的数量和学习率
            Shared_flag = getattr(args, 'Shared_flag', False)
            shared_mode_str = "Shared" if Shared_flag else "NonShared"

            # model_path = f'./saved_models/meta_policy_{shared_mode_str}_Innerlr{args.inner_lr}_Melr{args.meta_lr}_HIDDEN_DIM{args.HIDDEN_DIM}_train_NumTask{args.train_task_num}_epoch{args.num_epochs}.pth'
            model_path = f'./saved_models/meta_policy_Test_Innerlr{args.inner_lr}_Melr{args.meta_lr}_HIDDEN_DIM{args.HIDDEN_DIM}_train_NumTask{args.train_task_num}_epoch{args.num_epochs}.pth'
            if meta_learner.load_model(model_path, env=env):
                print("Model load success")
                report = check_model_parameters(meta_learner.meta_policy)

                if report["status"] == "OK":
                    print("模型参数检查通过，模型加载正常。")
                elif "WARNING" in report["status"]:
                    print("模型参数存在异常值，请人工复核。")
                else:
                    print("模型损坏，建议重新训练或检查文件。")
            else:
                print("model start training")
                meta_learner.meta_train()

            if alg =="Meta":
                Meta_alg = MetaOptimization(env, args, meta_learner)
            elif alg in ["MEBO","random_BO", "zero_BO", "standard_prior_BO"]:
                MEBO_alg = MetaLearningBayesianOptimizer(env, args, meta_learner,alg)
            print("Meta-training completed!")
            print("="*50 + "\n")

        elif alg == "Greedy":
            greedy_alg = GreedyCacheOptimizer(env, args, all_requests)
   
        elif alg == "Meta_OCO":
            # 创建元学习缓存算法
            Meta_OCO = MetaOCO(
                env, all_requests, request_records, cloud_manager.num_specific, cloud_manager.num_shared, NUM_MODELS, args)
        
        elif alg == "LRU":
            LRU_alg = LRUCacheOptimizer(env, args, env.cloud_manager.dependency_matrix, all_requests)
        
        elif alg == "Random":
            Random_alg = RandomCachingPlacement(env, args)
        elif alg == "MPUTA":
            MPUTA = MPUTADeployment(env, all_requests, request_records, cloud_manager.num_specific, cloud_manager.num_shared, NUM_MODELS, args)
        elif alg in ["random_BO", "zero_BO", "standard_prior_BO"]:
            print("This is used to test offline performance")
        else:
            print("The alg has not found")

        current_time = int(train_end_time)  # 从给定的结束时间开始
        online_task_num = args.task_num
        # 检查任务数量是否超出时间范围
        if args.time_slot < (online_task_num * args.task_length) + train_end_time:
            print("定义的task数量超出了所有请求数量的定义")
            # 应该调整online_task_num而不是直接break
            online_task_num = (args.time_slot - train_end_time) // args.task_length
            print(f"调整任务数量为: {online_task_num}")

        # 初始化全局性能指标列表（收集每个时隙的原始数据）
        all_hit_rates = []
        all_accuracies = []
        all_cloud_access_rates = []
        all_total_costs = []

        all_switching_costs = []
        all_communication_costs = []
        all_update_costs = []
        all_inference_costs = []
        all_accuracy_costs = []
        
        env.reset()
        print("\nStarting online deployment...")
        print(f"\n=== Online test, Alg is {alg}")
        for task_id in range(online_task_num):  
            # 确保当前时间不超过总时间槽
            if current_time >= args.time_slot:
                break
                
            # 计算当前任务的结束时间
            task_end_time = min(current_time + args.task_length, args.time_slot)
      
            # 初始化任务性能指标
            task_hit_rates = []
            task_accuracies = []
            task_cloud_access_rates = []
            task_total_costs = []
            task_switching_costs = []
            task_communication_costs = []
            task_update_costs = []
            task_inference_costs = []
            task_accuracy_costs =[]

            print(f"\n=== 任务数 {task_id+1}/{online_task_num} ===")

            for t in range(current_time, task_end_time):
                # 获取当前时间步的请求
                requests = all_requests[t]
                update_mask = 0
                iter_start = time.time()
                
                if alg in ["MEBO", "random_BO", "zero_BO", "standard_prior_BO"]:
                    new_cache_state, update_mask =  optimized_params, value = MEBO_alg.optimize(t, requests, env.cache_state)
                elif alg == "Greedy":
                    new_cache_state, update_mask = greedy_alg.greedy_optimize(t)
                elif alg == "Random":
                    new_cache_state, update_mask = Random_alg.update_cache(env.cache_state)
                elif alg == "Meta_OCO":
                    new_cache_state, update_mask = Meta_OCO.optimize(t, requests)
                elif alg == "LRU":
                    new_cache_state, update_mask = LRU_alg.lru_optimize(t)
                elif alg == "MPUTA":
                    new_cache_state, update_mask = MPUTA.optimize(t, requests, env.cache_state)
                elif alg == "Meta":
                    new_cache_state, update_mask = Meta_alg.optimize(t, requests, env.cache_state)
                else:
                    print("算法不存在")

                # 更新环境中的状态
                env.pre_cache_state = env.cache_state.copy()
                env.cache_state = new_cache_state.copy()
                env.update_accuracies(t, update_mask, env.pre_cache_state)
                # 评估当前时间步的性能
                try:
                    (
                        hit_rate, avg_accuracy, cloud_access_rate,
                        total_cost, switching_cost, communication_cost,
                        update_cost, inference_cost, accuracy_cost
                    ) = env.evaluate_performance(t, requests, env.cloud_manager.dependency_matrix, prev_deployment = env.pre_cache_state, update_decision = update_mask)
                except Exception as e:
                    print(f"性能评估失败: {str(e)}")
                    # 创建安全的默认值
                    hit_rate = avg_accuracy = cloud_access_rate = 0
                    total_cost = switching_cost = communication_cost = 0
                    update_cost = inference_cost = accuracy_cost = 0

                # 收集当前时间步的性能指标（任务级别）
                task_hit_rates.append(hit_rate)
                task_accuracies.append(avg_accuracy)
                task_cloud_access_rates.append(cloud_access_rate)
                task_total_costs.append(total_cost)
                task_switching_costs.append(switching_cost)
                task_communication_costs.append(communication_cost)
                task_update_costs.append(update_cost)
                task_inference_costs.append(inference_cost)
                task_accuracy_costs.append(accuracy_cost)

                # 收集当前时间步的性能指标（全局）
                all_hit_rates.append(hit_rate)
                all_accuracies.append(avg_accuracy)
                all_cloud_access_rates.append(cloud_access_rate)
                all_total_costs.append(total_cost)
                all_switching_costs.append(switching_cost)
                all_communication_costs.append(communication_cost)
                all_update_costs.append(update_cost)
                all_inference_costs.append(inference_cost)
                all_accuracy_costs.append(accuracy_cost)

            # 更新当前时间
            current_time = task_end_time

            # 计算任务的平均性能
            if task_hit_rates:  # 确保列表不为空
                task_avg_hit_rate = np.mean(task_hit_rates)
                task_avg_accuracy = np.mean(task_accuracies)
                task_avg_cloud_access = np.mean(task_cloud_access_rates)
                task_total_cost = np.sum(task_total_costs)
                task_total_switching_cost = np.sum(task_switching_costs)
                task_total_communication_cost = np.sum(task_communication_costs)
                task_total_update_cost = np.sum(task_update_costs)
                task_total_inference_cost = np.sum(task_inference_costs)
                task_total_accuracy_cost = np.sum(task_accuracy_costs)
            else:
                task_avg_hit_rate = task_avg_accuracy = task_avg_cloud_access = 0
                task_total_cost = task_total_switching_cost = task_total_communication_cost = 0
                task_total_update_cost = task_total_inference_cost = task_total_accuracy_cost = 0

            # 存储任务性能
            performance_history.append((
                task_avg_hit_rate, task_avg_accuracy, task_avg_cloud_access,
                task_total_cost, task_total_switching_cost, task_total_communication_cost,
                task_total_update_cost, task_total_inference_cost, task_total_accuracy_cost
            ))
            print(f"任务性能指标: 平均命中率={task_avg_hit_rate:.4f}, 平均精度={task_avg_accuracy:.4f}, 平均云访问率={task_avg_cloud_access:.4f}")
            print(f"任务成本指标: 总成本={task_total_cost:.2f}, 切换={task_total_switching_cost:.2f}, 通信={task_total_communication_cost:.2f}, 更新={task_total_update_cost:.2f}, 推理={task_total_inference_cost:.2f}, 精度成本={task_total_accuracy_cost:.2f}\n")

        # ====================== 最终性能分析 ======================
        if performance_history:
            total_time = time.time() - start_time
            
            # 基于任务平均的指标
            task_based_avg_hit_rate = np.mean([p[0] for p in performance_history])
            task_based_avg_accuracy = np.mean([p[1] for p in performance_history])
            task_based_avg_cloud_access = np.mean([p[2] for p in performance_history])
            task_based_total_cost = sum([p[3] for p in performance_history])
            task_based_total_switching_cost = sum([p[4] for p in performance_history])
            task_based_total_communication_cost = sum([p[5] for p in performance_history])
            task_based_total_update_cost = sum([p[6] for p in performance_history])
            task_based_total_inference_cost = sum([p[7] for p in performance_history])
            task_based_total_accuracy_cost = sum([p[8] for p in performance_history])
            
            # 基于时隙平均的指标
            slot_based_avg_hit_rate = np.mean(all_hit_rates)
            slot_based_avg_accuracy = np.mean(all_accuracies)
            slot_based_avg_cloud_access = np.mean(all_cloud_access_rates)
            slot_based_total_cost = sum(all_total_costs)
            slot_based_total_switching_cost = sum(all_switching_costs)
            slot_based_total_communication_cost = sum(all_communication_costs)
            slot_based_total_update_cost = sum(all_update_costs)
            slot_based_total_inference_cost = sum(all_inference_costs)
            slot_based_total_accuracy_cost = sum(all_accuracy_costs)
            
            print(f"\n=== 所有任务完成 ===")
            print(f"基于任务平均的总体性能指标:")
            print(f"  平均命中率={task_based_avg_hit_rate:.4f}, 平均精度={task_based_avg_accuracy:.4f}, 平均云访问率={task_based_avg_cloud_access:.4f}")
            print(f"  总成本={task_based_total_cost:.2f}, 切换={task_based_total_switching_cost:.2f}, 通信={task_based_total_communication_cost:.2f}, 更新={task_based_total_update_cost:.2f}, 推理={task_based_total_inference_cost:.2f}, 精度成本={task_based_total_accuracy_cost:.2f}")
            
            print(f"\n基于时隙平均的总体性能指标:")
            print(f"  平均命中率={slot_based_avg_hit_rate:.4f}, 平均精度={slot_based_avg_accuracy:.4f}, 平均云访问率={slot_based_avg_cloud_access:.4f}")
            print(f"  总成本={slot_based_total_cost:.2f}, 切换={slot_based_total_switching_cost:.2f}, 通信={slot_based_total_communication_cost:.2f}, 更新={slot_based_total_update_cost:.2f}, 推理={slot_based_total_inference_cost:.2f}, 精度成本={slot_based_total_accuracy_cost:.2f}")
            
            print(f"\n总耗时: {total_time:.2f}秒")
        else:
            print("没有任务完成")

        # ===== 保存到Excel文件(同时包含最终结果和详细历史) =====
        try:
            os.makedirs(f"./results/{alg}", exist_ok=True)

            # 获取两个新标识（默认为False）
            Update_flag = getattr(args, 'Update_flag', False)
            Shared_flag = getattr(args, 'Shared_flag', False)

            # 将布尔值转为简短字符串，用于文件名
            update_str = "_Update" if Update_flag else "_NoUpdate"
            shared_str = "_Shared" if Shared_flag else "_NoShared"

            if alg in ["MEBO", "random_BO", "zero_BO", "standard_prior_BO"]:
                excel_filename = (
                    f"./results/{alg}/{alg}_TaksNum_{args.task_num}"
                    f"_metalr{args.meta_lr}_inner_lr{args.inner_lr}"
                    f"_init_points{args.init_points}_task_length_{args.task_length}"
                    f"{update_str}{shared_str}.xlsx"
                )
            else:
                # 其他算法文件名
                excel_filename = (
                    f"./results/{alg}/{alg}_TaksNum_{args.task_num}"
                    f"_task_length_{args.task_length}"
                    f"{update_str}{shared_str}.xlsx"
                )

            # 保存Excel文件
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # 工作表1: 最终结果
                final_results = pd.DataFrame({
                    '指标类型': ['基于任务平均']*9 + ['基于时隙平均']*9,
                    '指标': [
                        '平均命中率','平均模型精度','平均云访问率',
                        '总成本','切换成本','通信成本','更新成本','推理成本','精度成本',
                        '平均命中率','平均模型精度','平均云访问率',
                        '总成本','切换成本','通信成本','更新成本','推理成本','精度成本',
                    ],
                    '值': [
                        f"{task_based_avg_hit_rate:.4f}", f"{task_based_avg_accuracy:.4f}", f"{task_based_avg_cloud_access:.4f}",
                        f"{task_based_total_cost:.2f}", f"{task_based_total_switching_cost:.2f}", f"{task_based_total_communication_cost:.2f}",
                        f"{task_based_total_update_cost:.2f}", f"{task_based_total_inference_cost:.2f}", f"{task_based_total_accuracy_cost:.2f}",
                        f"{slot_based_avg_hit_rate:.4f}", f"{slot_based_avg_accuracy:.4f}", f"{slot_based_avg_cloud_access:.4f}",
                        f"{slot_based_total_cost:.2f}", f"{slot_based_total_switching_cost:.2f}", f"{slot_based_total_communication_cost:.2f}",
                        f"{slot_based_total_update_cost:.2f}", f"{slot_based_total_inference_cost:.2f}", f"{slot_based_total_accuracy_cost:.2f}",
                    ]
                })
                final_results.to_excel(writer, sheet_name='最终结果', index=False)
                
                # 工作表2: 任务性能历史
                task_history_data = []
                for t, perf in enumerate(performance_history):
                    task_history_data.append({
                        '任务ID': t+1,
                        '命中率': perf[0],
                        '平均精度': perf[1],
                        '云访问率': perf[2],
                        '总成本': perf[3],
                        '切换成本': perf[4],
                        '通信成本': perf[5],
                        '更新成本': perf[6],
                        '推理成本': perf[7],
                        '精度成本': perf[8],
                    })
                task_history_df = pd.DataFrame(task_history_data)
                task_history_df.to_excel(writer, sheet_name='任务性能历史', index=False)
                
                # 工作表3: 时隙性能数据
                slot_history_data = []
                for t, (hit, acc, cloud, tot, sw, comm, upd, inf, acc_c) in enumerate(zip(
                    all_hit_rates, all_accuracies, all_cloud_access_rates,
                    all_total_costs, all_switching_costs, all_communication_costs,
                    all_update_costs, all_inference_costs, all_accuracy_costs
                )):
                    slot_history_data.append({
                        '时隙ID': t+1,
                        '命中率': hit,
                        '平均精度': acc,
                        '云访问率': cloud,
                        '总成本': tot,
                        '切换成本': sw,
                        '通信成本': comm,
                        '更新成本': upd,
                        '推理成本': inf,
                        '精度成本': acc_c,
                    })
                slot_history_df = pd.DataFrame(slot_history_data)
                slot_history_df.to_excel(writer, sheet_name='时隙性能数据', index=False)
                
                # 设置第一个工作表为可见
                workbook = writer.book
                for sheet in workbook.worksheets:
                    sheet.sheet_view.tabSelected = True
                    break
            
            print(f"\nExcel文件已保存: {excel_filename}")
            print("包含工作表: 最终结果 | 任务性能历史 | 时隙性能数据")
            
        except Exception as e:
            print(f"\n保存Excel时出错: {str(e)}")
            print("已保留文本格式的性能日志: performance_log.txt")

        performance_history = []
        env.reset()

if __name__ == "__main__":
    main_simulation()
