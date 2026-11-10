# Qwen3.6-27B W4A8（w3_g4）端到端复刻指南：从量化权重到 vllm-ascend 服务

> 本分支 `feat/w4a8-dense-linear-v0.30.0` = **vllm-ascend v0.30.0 + ModelSlim W4A8 dense Linear 支持**（cherry-pick 自上游 [#20469](https://github.com/vllm-project/vllm-ascend/pull/20469)，commit `2697e7f`）。
>
> 配套量化仓（含校准集/配方/脚本/补丁）：**https://github.com/kunpengW-code/msmodelslim**
>
> 两个仓配合即可完整复刻 w3_g4 权重（24G），并复现精度与性能。

## 0. 交付成绩（复现期望值）

| 指标 | 数值 | 备注 |
|---|---|---|
| 体积 | **24G**（7 shards） | 源模型 FP 54G，W8A8 anchor 30G |
| AIME2024 / 2025 | **91.67 / 91.67**（两轮均值） | 两轮分别为 90.0/93.33 与 93.33/90.0，历史新高 |
| GPQA | 80.8（81.31 / 80.30） | |
| CEval macro | 84.5（84.46 / 84.59） | 1346 题 |
| 性能 vs W8A8 anchor | TPOT 4/5 场景更优（-1.4% ~ **-9.3%**），吞吐 3/5 场景 +4~10% | 同机同日 mtp K3 双模型对照 |

配方概要：语言层 64 层 = 37 层 gptq-W4A8 + 27 块 gptq-W8A8（首 10 层 + 中部敏感 16 层 + 末层）+ visual 27 块 minmax-W8A8 + MTP gptq-W4A8 + lm_head FLOAT；per-channel；**长链混合校准 268k token**（58 条短 CoT + 6 条 W8A8 参照模型亲笔 AIME 正确轨迹，这是 w3_g3→w3_g4 的唯一关键变量）。

## 1. 环境

| 项 | 量化 | 服务（本分支） |
|---|---|---|
| 硬件 | 8×910B4 32GB，单卡即可 | TP2/DP2 共 4 卡（或 TP4） |
| 容器镜像 | `quay.io/ascend/vllm-ascend:v0.27.1rc1` | `quay.io/ascend/vllm-ascend:v0.30.0` |
| 代码 | msmodelslim 个人仓（editable 安装） | **本分支** |
| 耗时 | ~7h（64 层 × ~6.7 min，gptq Hessian 主导） | 加载 + 图捕获 ~10 min |

源模型：FP 权重 `Qwen3.6-27B`（54G）。

## 2. 第一步：量化出权重（msmodelslim 仓）

详细版见 msmodelslim 仓 `w4a8_repro/README.md`（含校准集格式、补丁原理、全部坑），核心步骤：

```bash
# 1) 取配套仓，量化容器内 editable 安装
git clone https://github.com/kunpengW-code/msmodelslim
docker exec <量化容器> pip install -e /path/to/msmodelslim

# 2) 按仓内 w4a8_repro/ 布局放置资产（calib/configs/scripts），注意 index.jsonl 内为绝对路径，跨机需 sed
# 3) 启动（参数必须裸名，脚本自己拼 configs/ 前缀）
docker exec <量化容器> bash -c \
  "cd /home/w30022218/w4a8_tuning && bash scripts/run_quant_arm_new.sh w3_g4.yaml w3_g4 5"
# 等价于 msmodelslim quant --model_path <FP模型> --save_path quant_out/w3_g4 \
#   --device npu --model_type Qwen3.6-27B --config_path configs/w3_g4.yaml --trust_remote_code True
```

产物：`quant_out/w3_g4/`（7 shards，24G，含 quant_model_description.json）。

量化侧三个必需补丁**已提交在 msmodelslim 仓 git 历史里**，无需手动打：
- `fix(ir)`：deployed fake-quant 包装器尾接 `.to(x.dtype)`（否则 gptq fp32 Hessian 泄漏撞 GDN conv1d，30 秒崩）
- `fix(qwen3_5_moe)` sdpa：eager→sdpa（否则 20k+ token 校准样本注意力矩阵 30.24 GiB OOM）
- `fix(mtp)`：lm_head 只算最后位置（否则 [1,20k,vocab] fp32 logits 24 GiB OOM）

⚠️ `save_path` 不复用：目录已存在会异常，换新名或清空。

## 3. 第二步：在 vllm-ascend 上运行（本分支）

**为什么需要本分支**：stock v0.30.0 只给 FusedMoE 注册了 `W4A8_DYNAMIC`，dense Linear 层直接 `NotImplementedError`（`modelslim_config.py` 走 `get_scheme_class` 注册表查不到即抛）。本分支补齐 `AscendW4A8DynamicLinearMethod`（A8W4 GMM、NZ 布局、int32 重打包、TP scale_bias 16 段分片）。

### 3.1 部署本分支（二选一）

**方式 A：容器内 pip install（标准，完整编译）**

```bash
git clone -b feat/w4a8-dense-linear-v0.30.0 https://github.com/kunpengW-code/vllm-ascend.git
docker cp vllm-ascend <v0.30.0容器>:/tmp/
docker exec <容器> bash -c "cd /tmp/vllm-ascend && pip install . --no-build-isolation"
```

**方式 B：PYTHONPATH + 嫁接编译产物（快，实测验证过）**

> ⚠️ 坑：v0.30.0 镜像内 vllm_ascend 是「源码 + 编译产物」形态（`/vllm-workspace/vllm-ascend`），
> `aclnnAddRmsNormBias` 等 fusion pass 依赖的 **vendor 自定义算子在编译产物里，不在宿主 CANN**。
> 纯源码 checkout 用 PYTHONPATH 遮蔽镜像内构建，会在 `norm_quant_fusion_pass` 注册阶段崩
> `RuntimeError: aclnnAddRmsNormBias ... not in libopapi.so`。嫁接产物即可：

```bash
git clone -b feat/w4a8-dense-linear-v0.30.0 https://github.com/kunpengW-code/vllm-ascend.git
cp -a /vllm-workspace/vllm-ascend/vllm_ascend/{_build_info.py,_version.py,include,lib, \
  libvllm_ascend_kernels.so,vllm_ascend_C.*.so,_cann_ops_custom/vendors} vllm-ascend/vllm_ascend/
# 服务时前置 PYTHONPATH（追加式，保住原 Ascend 路径）
export PYTHONPATH=/path/to/vllm-ascend:$PYTHONPATH
```

（同为 v0.30.0 基座，ABI 兼容；产物是构建输出，不入 git。）

### 3.2 启动服务（评测口径，逐字使用）

```bash
vllm serve quant_out/w3_g4 \
  --served-model-name w3g4 --host 127.0.0.1 --port <port> \
  --tensor-parallel-size 2 --data-parallel-size 2 --dtype bfloat16 \
  --language-model-only --safetensors-load-strategy prefetch \
  --no-enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --max-model-len 131072 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,24,32,48,64]}' \
  --seed 0 --kv-cache-dtype bfloat16
```

> ⚠️ **勿加额外旗标**：quantization 由 checkpoint 自带 quant_config 自动识别（引擎日志 `quantization=ascend`），不要显式 `--quantization ascend`；也不要 `--mamba-cache-mode align`。实测 TP4 + 这两个旗标的组合会在图编译阶段崩（宿主 CANN 缺 `aclnnAddRmsNormBias`）。

### 3.3 冒烟验证

```bash
curl -s http://127.0.0.1:<port>/health                        # 200
curl -s http://127.0.0.1:<port>/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"w3g4","messages":[{"role":"user","content":"1+1等于几？"}],"max_tokens":8,"temperature":0}'
```

## 4. 第三步：精度复现

工具链：msmodelslim 仓 `w4a8_repro/eval/`（eval2_client.py + run_eval2.sh + driver9.sh + CEval 套件）。

协议（与两轮全量严格一致）：
- **AIME2024/2025**：t=1.0 / top_p=0.95 / top_k=20 / seed=0 / max 65536 / batch 16
- **GPQA**：t=0.6 / top_p=0.95 / seed=0 / max 32768 / batch 32
- 评分：AIME 取最后一个 `\boxed{}`（gold 为字符串需 norm_int）；GPQA 取最后一个 "Answer: X"

期望（两轮）：AIME2024 90.0/93.33 → **91.67**；AIME2025 93.33/90.0 → **91.67**；GPQA 81.31/80.30 → **80.8**；CEval 84.46/84.59 → **84.5**；长链哨兵样本 id=22 两轮均收敛（不收敛即长程退化复发）。

## 5. 第四步：性能复现

框架：PerfTest（5 场景 × 2 轮 × 384 请求，mtp K3 / TP4）。基线用 **W8A8 anchor 同机同日对照**（W8A8 模型 30G 可另行获取）。

参照结论（w3_g4 vs W8A8 anchor）：TPOT S3 **-9.3%** / S5 **-8.6%**、其余 2 场景 -1.4% 上下；吞吐 3/5 场景 +4~10%；长 prompt 冷 prefill 略慢 3~7%。注意口径：框架自带 baseline.json 为 W8A8+DSpark K7 混合口径，仅作参考线；"S1 E2E +18%"一类数字需核对生成长度（~4200 vs ~3500 tok 假象）。

## 6. 已知坑速查

| 坑 | 解法 |
|---|---|
| `msmodelslim quant` 参数 | 配置名**裸名**（不带路径），脚本内部拼前缀 |
| 容器内 heredoc 打补丁 | `docker exec` 必须加 `-i`，否则 stdin 不转发且静默 rc=0 |
| 源码 checkout 服务崩溃 | 见 §3.1 方式 B，嫁接编译产物 |
| serve 加了额外旗标崩溃 | 见 §3.2，逐字用评测口径 |
| 杀 serve 残留 | vllm 子进程被 setproctitle 改名 `VLLM::DPCoordinator`/`VLLM::APIServer_x`，`grep python/vllm` 漏杀；看容器全量 ps 认 `VLLM::` 前缀 |
| AIME 双年连跑 ~1.5-2h 偶发服务器猝死 | 基建问题；死轮签名 = predictions 含 `"error":` ≥10，`--only` 补测即可 |

## 7. 验证记录

- 2026-11-09，131 服务器：本分支 PYTHONPATH 直服务（无 sitecustomize 注入），TP2/DP2 + 嫁接产物，`HEALTH_OK` + 真实补全返回；engine 进程 `/proc` 实锤加载分支代码。模型加载 22.4s。
- 分支提交 `1fbafb271` = 上述验证时的 commit（cherry-pick 零冲突 + Signed-off-by）。
