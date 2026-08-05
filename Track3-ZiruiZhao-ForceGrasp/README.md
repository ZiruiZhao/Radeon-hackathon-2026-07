# GenesisPick-VLA: End-to-End Robot Manipulation on AMD Radeon GPU / ROCm

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-red.svg)](https://rocm.docs.amd.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-orange.svg)](https://pytorch.org)
[![Genesis](https://img.shields.io/badge/Genesis-Physics-green.svg)](https://github.com/Genesis-Embodied-AI/Genesis)

**AMD AI DevMaster Hackathon 2026 — Track 3: Physical AI**  
**Team**: FruitNinja (赵子睿) · Solo Participant

> [中文版本](#chinese-version) | [English Version](#english-version)

---

## English Version

### 1. Project Introduction

**Problem**: Robot manipulation learning suffers from a critical data bottleneck. Collecting real-world demonstration data is expensive, slow, and hard to scale. Synthetic data generation in physics simulators offers a solution — but the fidelity of the physics, particularly how grasping is simulated, fundamentally determines whether the learned policy actually works.

**Solution**: This project implements and systematically compares two end-to-end robot learning pipelines on a single AMD Radeon GPU with ROCm:

- **Paradigm A (Position-Controlled Grasping)**: IK trajectory planning with position-commanded gripper fingers. Fast to implement and computationally efficient, but produces physically unrealistic grasping behavior.
- **Paradigm B (Force-Controlled Grasping)**: Force-based gripper closure with contact-aware grasping and velocity-limited transport. Physically accurate but computationally more demanding.

**Key Findings**: Position-controlled training data produces policies that achieve 70%+ in self-consistent evaluation but catastrophically drop to ~15% when tested under real physics. Force-controlled data, while harder to generate, produces policies that genuinely learn contact dynamics.

**Core Metrics**:
| Metric | Paradigm A (Position) | Paradigm B (Force) | Paradigm B+ (Force+DR) |
|--------|----------------------|-------------------|----------------------|
| Eval Success (self-consistent) | 75% | 45% | 55% |
| Eval Success (cross-paradigm) | 15% | 65% | 55% |
| Training Time | 7 min | 7 min | 11 min |
| Peak VRAM | 2.3 GB | 2.3 GB | 2.5 GB |

**Tech Stack**: Genesis (physics simulation) + LeRobot (dataset) + SmolVLA (450M VLA model) + PyTorch ROCm + radeonsi rendering, all running on AMD Radeon PRO R9700 / W7900D.

---

### 2. Development Process

#### 2.1 Key Decisions

1. **Why compare two paradigms instead of just improving one?** Our initial implementation used position-controlled grasping, but experiments revealed this produces what is essentially "fake" grasping — the cube stays in the gripper only because the fingers act as a scoop, not because any real contact forces are involved. The cube is not physically constrained; it merely rests on the fingertips. We decided that systematically quantifying and exposing the gap between position-commanded and force-controlled grasping would be more scientifically valuable than merely tuning hyperparameters on the simpler approach.

2. **Why franka_fruit_pick for Paradigm B?** This open-source implementation provides a proven force-controlled grasping pipeline (`control_dofs_force`) with velocity-limited transport and per-object grasp profiles. Building on this allowed us to focus on the comparison analysis rather than reimplementing low-level grasping physics.

3. **Why cross-paradigm evaluation?** Standard evaluation only tests a policy in the same physics regime it was trained in. This is circular — a policy that learned to "cheat" appears successful. Cross-paradigm evaluation (train in A, test in B) reveals the true generalization capability.

#### 2.2 Difficulties Encountered

1. **SSH Access to Cloud Instance**: The Radeon Cloud container image does not include SSH server by default (FAQ Q40/Q78). Solution: install openssh-server manually via JupyterLab Terminal (`apt install -y openssh-server && service ssh start`), register SSH public key in Profile, and ensure the template has SSH Access enabled.

2. **Cube Position Range Modification**: The `scene_placement.py` script defines `CUBE_RANGE_X` and `CUBE_RANGE_Y` as internal constants rather than CLI arguments. Expanding the cube randomization range for better workspace coverage required modifying these constants directly to test the arm's full reachable workspace.

3. **Cross-Paradigm Evaluation Implementation**: Running Paradigm A's model in Paradigm B's physics environment required careful handling of Genesis scene initialization and camera layout compatibility.

#### 2.3 Future Improvements

- Extend to more objects (YCB dataset: banana, lemon, plum, mug) with object-specific grasp profiles
- Add domain randomization layers (appearance + physics)
- Implement sim-to-real transfer experiments on physical Franka hardware
- Explore multi-task VLA training across multiple manipulation primitives

---

### 3. Code Source Attribution

This project builds on the open-source robot learning ecosystem. All components are used in compliance with their respective licenses (MIT, Apache-2.0).

| Component | Source | Our Modifications |
|-----------|--------|-------------------|
| Data generation pipeline | Built on Genesis API + LeRobot framework | Extended cube position range, added cross-paradigm protocol |
| `02_train_vla.py` | LeRobot SmolVLA training example | Adapted for AMD ROCm environment |
| `04_eval_custom_scene.py` | LeRobot evaluation framework | Added cross-paradigm evaluation with force-controlled physics |
| `pick_common.py`, `genesis_scene_utils.py`, `scene_placement.py` | Genesis community utilities | Extended cube ranges, tuned camera positions |
| `real_physics_grasp.py` | **ORIGINAL** | Force-controlled grasping pipeline for kitchen scenes |
| `cloud_setup_and_run.sh` | **ORIGINAL** | Automated cloud execution pipeline |
| `TECHNICAL_REPORT.md` | **ORIGINAL** | Complete comparative analysis |
| SmolVLA model (450M) | HuggingFace LeRobot | Fine-tuned on our datasets |
| Genesis physics engine | Genesis-Embodied-AI | Used via public API |
| Franka Panda URDF | Genesis bundled assets | Unmodified |
| Kitchen scene mesh | Open-source GLB asset | Unmodified |

**Original Contributions**:
- Cross-paradigm evaluation protocol: train on one physics regime, test on another
- Quantitative analysis framework for physics fidelity in robot learning
- Force-controlled grasping adaptation for kitchen tabletop scenes
- End-to-end pipeline automation for AMD ROCm cloud instances

---

### 4. Team Contribution

**FruitNinja (赵子睿)** — Solo Participant (100% contribution)

- **Simulation Environment**: Set up Genesis + ROCm environment, kitchen scene configuration, camera layout
- **Data Pipeline**: Built and debugged both position-controlled and force-controlled data generation pipelines; modified cube position ranges for better coverage
- **Model Training**: Fine-tuned SmolVLA (450M) on AMD Radeon GPU with BF16 AMP + SDPA optimization
- **Evaluation**: Implemented closed-loop evaluation and cross-paradigm testing protocol
- **Analysis**: Quantified the impact of physics fidelity on policy learning; identified the "scooping" failure mode
- **Documentation**: Bilingual README, technical report, reproducibility guide
- **Video**: Planning and recording

---

### 5. Installation & Execution

#### 5.1 Environment Requirements

- AMD Radeon GPU with ROCm 7.x (RDNA4: R9700, or RDNA3.5: W7900)
- Ubuntu 22.04+ with Python 3.10–3.12
- Git LFS for large file handling

#### 5.2 Quick Setup (Cloud)

```bash
# Clone the repository
cd /workspace
git clone https://github.com/PhysicalAI-AIM/Robot_synthetic_data_generation_workshop.git
cd Robot_synthetic_data_generation_workshop

# Run the setup + execution script
bash cloud_setup_and_run.sh
```

#### 5.3 Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv rdna && source rdna/bin/activate

# 2. Install ROCm PyTorch (use prebuilt wheels from /torch-*.whl on Radeon Cloud)
pip install /torch-2.9.1+rocm7.2.1*.whl /torchvision-0.24.0+rocm7.2.1*.whl

# 3. Install Genesis from GitHub main branch
git clone --depth 1 https://github.com/Genesis-Embodied-AI/Genesis.git
pip install -e Genesis/

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Install FFmpeg
apt-get install -y ffmpeg

# 6. Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### 5.4 Run the Pipeline

```bash
# Paradigm A — Position-controlled data generation (100 episodes)
python scripts/02_gen_data_custom_scene.py \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 100 --seed 42 \
  --repo-id local/franka-kitchen-wrist-100ep

# Paradigm A — Training (4000 steps)
python scripts/02_train_vla.py \
  --dataset-id local/franka-kitchen-wrist-100ep \
  --n-steps 4000 --batch-size 4 --num-workers 4 \
  --run-name smolvla_kitchen_wrist

# Paradigm A — Evaluation
python scripts/04_eval_custom_scene.py \
  --checkpoint output/train/smolvla_kitchen_wrist/final \
  --dataset-id local/franka-kitchen-wrist-100ep \
  --scene rustic_kitchen --anchor floor_origin \
  --camera-layout up_wrist \
  --n-episodes 20 --seed 99 --record-video

# Paradigm B — Force-controlled grasping + fruit pick pipeline
# See cloud_setup_and_run.sh for the complete workflow
```

#### 5.5 Expected Output

- Training: loss decreases from ~0.67 to ~0.016 over 4000 steps
- Evaluation: success rate 40-75% (paradigm-dependent), evaluation videos in `output/eval/`
- Total pipeline runtime: ~34-40 minutes on AMD Radeon PRO R9700

---

## 中文版本

### 1. 项目简介

**问题**：机器人操作学习面临严重的数据瓶颈。收集真实世界的演示数据成本高、速度慢、难以规模化。物理仿真器中的合成数据生成提供了解决方案——但物理保真度，尤其是抓取如何被仿真，从根本上决定了学习到的策略是否真的有效。

**解决方案**：本项目在单张AMD Radeon GPU（ROCm）上实现并系统对比了两条端到端机器人学习流水线：

- **方案A（位置控制抓取）**：IK轨迹规划 + 位置指令控制夹爪手指——常见于大多数开源机器人学习管线。快速但物理上不真实。
- **方案B（力控制抓取）**：基于力的夹爪闭合 + 接触感知抓取 + 速度限制运输。物理准确但计算更密集。

**核心发现**：位置控制训练数据产生的策略在自洽评估中达到70%+，但在真实物理下测试时骤降至约15%。力控制数据虽然更难生成，但产生的策略真正学会了接触动力学。

**技术栈**：Genesis（物理仿真）+ LeRobot（数据集）+ SmolVLA（4.5亿参数VLA模型）+ PyTorch ROCm + radeonsi渲染，全部运行在AMD Radeon PRO R9700 / W7900D上。

### 2. 开发过程

#### 2.1 关键决策

1. **为什么对比两种方案？** 我们在实现过程中发现，位置控制抓取产生的实际上是"假"抓取——方块没有被真正夹住，只是靠手指托举。这个现象在机器人学习领域广泛存在但很少被量化讨论。我们决定系统量化两种抓取物理机制之间的差距，这比仅仅调参更有科学价值。

2. **为什么用franka_fruit_pick做方案B？** 这个参考实现方案已经实现了力控制抓取（`control_dofs_force`）、速度限制运输和逐物体抓取配置。直接使用节省开发时间并确保可复现性。

3. **为什么做跨方案评估？** 标准评估只在与训练相同的物理机制下测试策略。这是循环论证——学会了"作弊"的策略看起来是成功的。跨方案评估揭示了真正的泛化能力。

#### 2.2 遇到的困难

1. **SSH连接云实例**：Radeon Cloud容器镜像默认不含SSH Server。解决方案：通过JupyterLab Terminal手动安装openssh-server，在Profile中注册SSH公钥，并确保Template开启了SSH Access。

2. **修改方块位置范围**：工作坊脚本将`CUBE_RANGE_X`和`CUBE_RANGE_Y`定义为硬编码常量而非CLI参数。扩展范围需要直接修改源码文件`scene_placement.py`。

#### 2.3 未来改进

- 扩展到更多物体（YCB数据集：香蕉、柠檬、李子、杯子）配合逐物体抓取配置
- 添加domain randomization（外观+物理双层）
- 在真实Franka硬件上实现sim-to-real迁移实验
- 探索跨多个操作原语的多任务VLA训练

### 3. 代码来源说明

| 组件 | 来源 | 修改 |
|------|------|------|
| `02_gen_data_custom_scene.py` 等 | PhysicalAI-AIM工作坊 | 修改`scene_placement.py`中方块范围 |
| `real_physics_grasp.py` | **原创** — 受franka_fruit_pick启发 | 全新实现，结合厨房场景与力控制抓取 |
| `cloud_setup_and_run.sh` | **原创** | 完整的云端执行流水线脚本 |
| `TECHNICAL_REPORT.md` | **原创** | 包含对比分析的完整技术报告 |
| SmolVLA模型 | HuggingFace LeRobot | 微调，未修改架构 |
| Genesis物理引擎 | Genesis-Embodied-AI | 通过API使用，未修改 |

### 4. 团队分工

**赵子睿 (FruitNinja)** — 个人参赛（100%贡献）

仿真环境搭建、数据流水线构建与调试、模型训练与评估、跨方案分析、文档撰写、演示视频制作。

### 5. 安装与运行

参见 [English Version Section 5](#5-installation--execution) 了解详细的安装和运行命令。

#### 环境要求
- AMD Radeon GPU + ROCm 7.x
- Ubuntu 22.04+ / Python 3.10–3.12
- Genesis（GitHub main分支）+ LeRobot 0.4.4 + PyTorch ROCm版

#### 一键运行（云实例）
```bash
cd /workspace/Robot_synthetic_data_generation_workshop
bash cloud_setup_and_run.sh
```

---

## Known Issues

1. **ROCm Genesis import**: Genesis must be installed from the GitHub `main` branch, not PyPI 0.4.5, to avoid `cuda.bindings` import issues on ROCm.
2. **Camera layout consistency**: Training and evaluation MUST use the same `--camera-layout up_wrist` flag. Omitting it during eval loads a world-fixed side camera, causing observation space mismatch.
3. **W7900 lower success rate**: RDNA3.5 (W7900) shows ~12% eval success vs RDNA4 (R9700) ~48%, likely due to ROCm driver version differences (7.0.2 vs 7.2).
4. **TorchCodec is CPU-only**: Video decoding runs on CPU; SmolVLA training/inference still runs on GPU. This is expected and does not affect training performance.

## References

- [Genesis Physics Engine](https://github.com/Genesis-Embodied-AI/Genesis)
- [LeRobot: Robot Learning Platform](https://github.com/huggingface/lerobot)
- [Force-Controlled Franka Pick (Force-Controlled Grasping)](https://github.com/wangxunx/franka_fruit_pick)
- [AMD ROCm Documentation](https://rocm.docs.amd.com)
- [AMD-DEV-CONTEST/Radeon-hackathon-2026-07](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07)
