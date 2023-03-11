import gradio as gr
import os
from datetime import datetime
from huggingface_hub import hf_hub_download
from pynvml import *
nvmlInit()
gpu_h = nvmlDeviceGetHandleByIndex(0)
ctx_limit = 1848
desc = f'''链接：
<a href='https://github.com/BlinkDL/ChatRWKV' target="_blank" style="margin:0 0.5em">ChatRWKV</a>
<a href='https://github.com/BlinkDL/RWKV-LM' target="_blank" style="margin:0 0.5em">RWKV-LM</a>
<a href="https://pypi.org/project/rwkv/" target="_blank" style="margin:0 0.5em">RWKV pip package</a>
<a href="https://zhuanlan.zhihu.com/p/609154637" target="_blank" style="margin:0 0.5em">知乎教程</a>
'''

os.environ["RWKV_JIT_ON"] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # if '1' then use CUDA kernel for seq mode (much faster)

from rwkv.model import RWKV
model_path = hf_hub_download(repo_id="BlinkDL/rwkv-4-pile-7b", filename="RWKV-4-Pile-7B-EngChn-testNovel-1883-ctx2048-20230310.pth")
model = RWKV(model=model_path, strategy='cuda fp16')
from rwkv.utils import PIPELINE, PIPELINE_ARGS
pipeline = PIPELINE(model, "20B_tokenizer.json")

def infer(
        ctx,
        token_count=10,
        temperature=1.0,
        top_p=0.8,
        presencePenalty = 0.1,
        countPenalty = 0.1,
):
    args = PIPELINE_ARGS(temperature = max(0.2, float(temperature)), top_p = float(top_p),
                     alpha_frequency = countPenalty,
                     alpha_presence = presencePenalty,
                     token_ban = [0], # ban the generation of some tokens
                     token_stop = []) # stop generation whenever you see any token here

    ctx = ctx.strip(' ')
    if ctx.endswith('\n'):
        ctx = f'\n{ctx.strip()}\n'
    else:
        ctx = f'\n{ctx.strip()}'

    gpu_info = nvmlDeviceGetMemoryInfo(gpu_h)
    print(f'vram {gpu_info.total} used {gpu_info.used} free {gpu_info.free}')
    
    all_tokens = []
    out_last = 0
    out_str = ''
    occurrence = {}
    state = None
    for i in range(int(token_count)):
        out, state = model.forward(pipeline.encode(ctx)[-ctx_limit:] if i == 0 else [token], state)
        for n in args.token_ban:
            out[n] = -float('inf')
        for n in occurrence:
            out[n] -= (args.alpha_presence + occurrence[n] * args.alpha_frequency)

        token = pipeline.sample_logits(out, temperature=args.temperature, top_p=args.top_p)
        if token in args.token_stop:
            break
        all_tokens += [token]
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        
        tmp = pipeline.decode(all_tokens[out_last:])
        if '\ufffd' not in tmp:
            out_str += tmp
            yield out_str.strip()
            out_last = i + 1
    yield out_str.strip()

examples = [
    ["以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\n第一章", 200, 0.9, 0.85, 0.1, 0.1],
    ["“区区", 200, 0.9, 0.85, 0.1, 0.1],
    ["这是一个修真世界，详细世界设定如下：\n1.", 200, 0.9, 0.85, 0.1, 0.1],
    ["怎样创立一家快速盈利的AI公司：\n1.", 200, 0.9, 0.8, 0.1, 0.1],
    ["帝皇是一名极为强大的灵能者，而且还是永生者：一个拥有无穷知识与力量以及使用它们的雄心的不朽存在。根据传说，", 200, 0.9, 0.85, 0.1, 0.1],
    ["我问智脑：“三体人发来了信息，告诉我不要回答，这是他们的阴谋吗？”", 200, 0.9, 0.8, 0.1, 0.1],
    ["我问编程之神：“Pytorch比Tensorflow更好用吗？”", 200, 0.9, 0.8, 0.1, 0.1],
    ["这竟然是", 200, 0.9, 0.85, 0.1, 0.1],
    ["Translation Samples\nChinese: 修道之人，最看重的是什么？\nEnglish:", 200, 0.9, 0.5, 0.1, 0.1],
]

iface = gr.Interface(
    fn=infer,
    description=f'''{desc} <b>请点击例子（在页面底部）</b>，可以编辑内容。这里模型只看左边输入的最后约1100字，一定要写好，标点规范，无错别字，否则电脑也会学你的错误。为避免占用资源，每次生成限制长度。可将右边内容复制到左边，然后继续生成。''',
    allow_flagging="never",
    inputs=[
        gr.Textbox(lines=10, label="Prompt", value="以下是不朽的科幻史诗长篇巨著，描写细腻，刻画了数百位个性鲜明的英雄和宏大的星际文明战争。\n第一章"),  # prompt
        gr.Slider(10, 200, step=10, value=200, label="token_count，每次生成的长度"),  # token_count
        gr.Slider(0.2, 2.0, step=0.1, value=0.9, label="temperature，默认0.9，越高越标新立异，越低越循规蹈矩"),  # temperature
        gr.Slider(0.0, 1.0, step=0.05, value=0.85, label="top_p，默认0.85，越高越变化丰富，越低越循规蹈矩"),  # top_p
        gr.Slider(0.0, 1.0, step=0.1, value=0.1, label="presencePenalty，避免已经写过的字"),  # presencePenalty
        gr.Slider(0.0, 1.0, step=0.1, value=0.1, label="countPenalty，额外避免已经写过多次的字"),  # countPenalty
    ],
    outputs=gr.Textbox(label="Generated Output", lines=28),
    examples=examples,
    cache_examples=False,
).queue()

demo = gr.TabbedInterface(
    [iface], ["Generative"]
)

demo.queue()
demo.launch(share=False)
