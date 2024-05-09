# GroundingDINO Inference Export and Comparison

## 环境

python 3.8.10 \
pytorch 2.3.0+cu121

python 环境配置（在 GroundingDINO/ 目录下）：

```bash
$ pip install -r requirements.txt
$ pip install openvino==2023.1.0.dev20230811 openvino-dev==2023.1.0.dev20230811 onnx onnxruntime
$ pip install -e .
```

## Inference

将 GroundingDINO/groudingdino/models/GroundingDINO/ 目录下的 groudingdino.py 替换为 groundingdino.py.inference

执行命令（在 GroundingDINO/ 目录下）：

```bash
$ python3 demo/inference_on_a_image.py \
    -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
    -p weights/groundingdino_swinb_cogcoor.pth \
    -i images/img.jpeg \
    -t "Something to detect." \
    -o output
```

-c: config文件，SwinB：groundingdino/config/GroundingDINO_SwinB_cfg.py ；SwinT：groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p: 参数路径，SwinB：weights/groundingdino_swinb_cogcoor.pth ；SwinT：weights/groundingdino_swint_ogc.pth \
-i: 输入图像 \
-t: 输入文本 \
-o: 输出文件夹路径，生成文件 pred.jpg（输出图像）；raw_image.jpg（输入图像）\
--box_threshold: optional \
--text_threshold: optional \
--token_spans: optional

## Export

由于onnx模型转化需要修改原模型，因此将 GroundingDINO/groudingdino/models/GroundingDINO/ 目录下的 groudingdino.py 替换为 groundingdino.py.export

执行命令（在 GroundingDINO/ 目录下）：

```bash
$ python3 demo/export_openvino.py \
    -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
    -p weights/groundingdino_swinb_cogcoor.pth \
    -o onnx_models/
```

-c: config文件，SwinB：groundingdino/config/GroundingDINO_SwinB_cfg.py ；SwinT：groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p: 参数路径，SwinB：weights/groundingdino_swinb_cogcoor.pth ；SwinT：weights/groundingdino_swint_ogc.pth \
-o: 输出文件夹路径，生成文件 grounded.onnx（onnx模型）；grounded.bin  grounded.xml（模型信息）

## Comparison

将 GroundingDINO/groudingdino/models/GroundingDINO/ 目录下的 groudingdino.py 替换为 groundingdino.py.inference

执行命令（在 GroundingDINO/ 目录下）：

```bash
$ python3 onnx_inference.py \
    -c groundingdino/config/GroundingDINO_SwinB_cfg.py \
    -p weights/groundingdino_swinb_cogcoor.pth \
    -i images/img.jpeg \
    -t "Something to detect." \
    -o output
```

-c: config文件，SwinB：groundingdino/config/GroundingDINO_SwinB_cfg.py ；SwinT：groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p: 参数路径，SwinB：weights/groundingdino_swinb_cogcoor.pth ；SwinT：weights/groundingdino_swint_ogc.pth \
-i: 输入图像 \
-t: 输入文本 \
-o: 输出文件夹路径，生成文件 pred.jpg（输出图像） \
--box_threshold: optional \
--text_threshold: optional \
--token_spans: optional

该脚本在terminal上按顺序分别打印：原PyTorch模型和onnx模型的预测logits，原模型和onnx模型的预测boxes \
同时，用np.testing.assert_allclose(rtol=1e-03, atol=1e-05) 对比两者输出（157、158行），若不满足容忍度范围会抛出异常 \
最终输出的图像为onnx模型inference结果 \
