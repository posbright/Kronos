# model 包对外入口：导出三大核心类，供 `from model import ...` 使用
from .kronos import KronosTokenizer, Kronos, KronosPredictor

# 模型名称 -> 模型类 的映射表，便于按字符串名称动态获取类
model_dict = {
    'kronos_tokenizer': KronosTokenizer,  # K 线量化器
    'kronos': Kronos,                      # 自回归主模型
    'kronos_predictor': KronosPredictor    # 高层预测封装
}


def get_model_class(model_name):
    """根据模型名称返回对应的模型类；未注册时抛出 NotImplementedError。"""
    if model_name in model_dict:
        return model_dict[model_name]
    else:
        print(f"Model {model_name} not found in model_dict")
        raise NotImplementedError


