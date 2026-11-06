"""
动物情绪识别 - Web 界面 (V5.1)
启动: python app.py
访问: http://localhost:7860

默认使用三模型加权集成 (Ensemble)，
也支持切换到单模型预测。
"""
import gradio as gr
import predict
import os
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "results", "models")

# 预加载所有可用模型
MODEL_PATHS = {
    "🏆 加权集成 (推荐)": None,  # ensemble 特殊处理
    "ResNet-50": os.path.join(MODEL_DIR, "best_model_resnet50.pth"),
    "EfficientNet-B3": os.path.join(MODEL_DIR, "best_model_efficientnet_b3.pth"),
    "DenseNet-121": os.path.join(MODEL_DIR, "best_model_densenet121.pth"),
}

# val_acc 权重 (用于加权集成)
ENSEMBLE_WEIGHTS = {
    'resnet50': 95.33, 'efficientnet_b3': 95.78, 'densenet121': 95.56
}

# 缓存
model_cache = {}
_ensemble_cache = None


def get_ensemble():
    """懒加载集成模型"""
    global _ensemble_cache
    if _ensemble_cache is None:
        _ensemble_cache = predict.load_ensemble_with_weights(ENSEMBLE_WEIGHTS)
    return _ensemble_cache


def get_model(model_name):
    """获取或加载单模型（带缓存）"""
    if model_name in model_cache:
        return model_cache[model_name]

    model_path = MODEL_PATHS.get(model_name)
    if model_path and os.path.exists(model_path):
        model, device, input_size, class_names = predict.load_model(model_path)
        model_cache[model_name] = (model, device, input_size, class_names)
        return model_cache[model_name]
    return None


def classify_emotion(image, model_choice):
    """Gradio 回调：默认加权集成，也可选单模型"""
    if image is None:
        return "请上传一张图片", {}

    tmp_path = os.path.join(tempfile.gettempdir(), "gradio_upload.jpg")
    image.save(tmp_path)

    if model_choice.startswith("🏆"):
        # ====== 加权集成模式 ======
        try:
            models, weights = get_ensemble()
            result = predict.predict_ensemble(tmp_path, models, weights)
        except Exception as e:
            return f"集成加载失败: {e}", {}
    else:
        # ====== 单模型模式 ======
        model_data = get_model(model_choice)
        if model_data is None:
            return f"模型 '{model_choice}' 未找到，请先训练。", {}
        model, device, input_size, class_names = model_data
        result = predict.predict_image(tmp_path, model, device, input_size, class_names)

    if result is None:
        return "预测失败", {}

    label = f"{result['class']}  ({result['confidence'] * 100:.1f}%)"
    detail = {k: float(v) for k, v in result["probabilities"].items()}
    return label, detail


# ====== Gradio 界面 ======
with gr.Blocks(title="动物情绪识别 V5.1", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🐶🐱 动物情绪识别系统 V5.1

    上传一张猫或狗的图片，AI 将识别情绪：**Angry（愤怒）/ Sad（悲伤）/ Happy（开心）**

    🏆 **默认使用三模型加权集成，精度更高！**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="📷 上传动物图片")
            model_choice = gr.Dropdown(
                choices=list(MODEL_PATHS.keys()),
                value="🏆 加权集成 (推荐)",
                label="🧠 选择模型",
            )
            submit_btn = gr.Button("🔍 开始识别", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="🎯 识别结果")
            prob_output = gr.Label(label="📊 各类别置信度", num_top_classes=3)

    submit_btn.click(
        fn=classify_emotion,
        inputs=[image_input, model_choice],
        outputs=[label_output, prob_output],
    )

    gr.Markdown("""
    ---
    ### 📋 模型说明

    | 模型 | 参数量 | Test Acc |
    |:---|:---:|:---:|
    | **🏆 加权集成** | 41M (3合1) | **96.00%** |
    | ResNet-50 | 23.5M | 95.11% |
    | EfficientNet-B3 | 10.7M | 95.11% |
    | DenseNet-121 | 7.0M | 94.67% |

    > 加权集成原理：三模型分别预测 → 以验证集精度为权重，对 softmax 概率加权平均。
    > 不同架构 CNN 误差互补，集成后提升 ~1%（Fort et al., NeurIPS 2019）。
    """)


if __name__ == "__main__":
    print("=" * 50)
    print("  🐶 动物情绪识别 Web 服务 V5.1")
    print("  默认: 三模型加权集成 (96.00%)")
    print("  访问: http://localhost:7860")
    print("=" * 50)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
