import torch

def check_model_parameters(model, verbose=True, tol_mean=1e3, tol_std=1e3):
    """
    检查 PyTorch 模型参数是否正确加载、是否存在异常值或数值溢出。

    参数:
        model (torch.nn.Module): 要检查的模型对象
        verbose (bool): 是否打印详细信息
        tol_mean (float): 参数均值异常阈值
        tol_std (float): 参数标准差异常阈值

    返回:
        dict: 检测结果，包括是否正常、异常层、参数统计等信息
    """

    result = {
        "total_params": 0,
        "trainable_params": 0,
        "layers_checked": 0,
        "has_nan_or_inf": False,
        "abnormal_layers": [],
        "param_mean_range": (None, None),
        "param_std_range": (None, None),
        "status": "UNKNOWN"
    }

    means, stds = [], []
    for name, param in model.named_parameters():
        result["layers_checked"] += 1
        result["total_params"] += param.numel()
        if param.requires_grad:
            result["trainable_params"] += param.numel()

        # 检查 NaN / Inf
        if torch.isnan(param).any() or torch.isinf(param).any():
            result["has_nan_or_inf"] = True
            result["abnormal_layers"].append((name, "NaN/Inf"))
            if verbose:
                print(f"⚠️ {name:25s} contains NaN or Inf values")
            continue

        # 标量参数处理
        if param.numel() <= 1:
            val = param.item()
            means.append(val)
            stds.append(0.0)
            if verbose:
                print(f"{name:25s} | scalar param (value={val:.4f})")
            continue

        # 计算均值和标准差
        mean_val = param.mean().item()
        std_val = param.std(unbiased=False).item()
        means.append(mean_val)
        stds.append(std_val)

        # 检查异常范围
        if abs(mean_val) > tol_mean or std_val > tol_std:
            result["abnormal_layers"].append((name, f"mean={mean_val:.2e}, std={std_val:.2e}"))
            if verbose:
                print(f"⚠️ {name:25s} | abnormal: mean={mean_val:.2e}, std={std_val:.2e}")
        elif verbose:
            print(f"{name:25s} | mean={mean_val:.4f}, std={std_val:.4f}")

    # 统计范围
    if means and stds:
        result["param_mean_range"] = (min(means), max(means))
        result["param_std_range"] = (min(stds), max(stds))

    # 状态判断
    if result["has_nan_or_inf"]:
        result["status"] = "ERROR: NaN/Inf found"
    elif len(result["abnormal_layers"]) > 0:
        result["status"] = "WARNING: abnormal values detected"
    else:
        result["status"] = "OK"

    if verbose:
        print("\n=== Model Parameter Summary ===")
        print(f"Total parameters     : {result['total_params']:,}")
        print(f"Trainable parameters : {result['trainable_params']:,}")
        print(f"Checked layers       : {result['layers_checked']}")
        print(f"Mean range           : {result['param_mean_range']}")
        print(f"Std range            : {result['param_std_range']}")
        print(f"Model status         : {result['status']}")

    return result
