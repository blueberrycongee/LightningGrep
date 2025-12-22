"""
测试训练好的模型
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_model(model_path="outputs/sft_v1", base_model="Qwen/Qwen2.5-1.5B-Instruct"):
    print("=" * 50)
    print("加载模型...")
    print("=" * 50)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    
    print("模型加载完成!\n")
    
    # 从验证集加载测试用例（确保不在训练集中）
    import json
    with open("data/synthetic/raw_5k_val.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # 取验证集的前 3 条（SFT 格式是 "input" 字段）
    test_queries = []
    for d in val_data[:3]:
        if "input" in d:
            # SFT 格式: "Query: xxx"
            q = d["input"].replace("Query: ", "")
        elif "question" in d:
            q = d["question"]
        else:
            q = str(list(d.values())[0])
        test_queries.append(q)
    print(f"从验证集加载 {len(test_queries)} 条测试数据\n")
    
    for i, query in enumerate(test_queries):
        print(f"{'='*50}")
        print(f"测试 {i+1}: {query}")
        print(f"{'='*50}")
        
        # 构建输入
        messages = [
            {"role": "user", "content": f"Query: {query}"}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        print(f"\n模型输出:\n{response}\n")


if __name__ == "__main__":
    test_model()
