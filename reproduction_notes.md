# 复现记录（micrograd）

日期：2026-10-05
仓库：`micrograd`（karpathy/micrograd），分支 `master`，基线提交 `7bc720e`
本文档记录当天在本机的完整操作过程，便于在另一台机器上复现。

---

## 1. 环境信息

| 项目 | 值 |
| --- | --- |
| 操作系统 | Microsoft Windows 11 专业版，64-bit（Build 10.0.26200） |
| 平台标识 | `Windows-10-10.0.26200-SP0`（`platform.platform()`） |
| CPU | Intel(R) Core(TM) i9-14900HX |
| Shell | Windows PowerShell 5.1.26100.9168 |
| 仓库根目录 | `C:\Users\Administrator\Desktop\学习\AI\Studying Record\10\10.04-AI Program\micrograd` |
| 虚拟环境路径 | `C:\Users\Administrator\Desktop\学习\AI\Studying Record\10\10.04-AI Program\micrograd\.venv` |
| Python 解释器 | `.\.venv\Scripts\python.exe` |
| Python 版本 | 3.9.25（`main, Nov 3 2025`，MSC v.1929 64 bit AMD64） |
| venv 基础环境 | `D:\Anaconda\envs\pytorch`（`sys.base_prefix`） |
| venv 配置文件 | `.venv\pyvenv.cfg`，其中 `include-system-site-packages = true` |
| micrograd 安装方式 | 可编辑安装（editable），版本 `0.1.0` |

说明：该 venv 通过 `--system-site-packages` 方式挂在 Anaconda 的 `pytorch` 环境上，
因此可以直接复用基础环境里已装好的第三方库，不需要在一个干净的 venv 里从头装一遍。

当天实际使用的关键依赖版本：

| 包 | 版本 |
| --- | --- |
| numpy | 2.0.1 |
| scipy | 1.13.1 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.9.2 |
| pytest | 8.4.2 |
| nbconvert | 7.17.1 |
| nbclient | 0.10.2 |
| ipykernel | 6.30.1 |
| jupyter / notebook / jupyterlab | 1.1.1 / 7.5.8 / 4.5.11 |
| torch（仅基础环境自带，本 demo 未使用） | 2.8.0+cu126 |

注意：`python`、`pip` 等命令**不在 PATH 上**，直接敲 `python` 会被 Windows 的
“应用执行别名”拦到 Microsoft Store。所有命令都必须显式使用 venv 里的解释器，
下文统一写作 `.\.venv\Scripts\python.exe`。

---

## 2. 安装命令

本次没有新装任何依赖，环境本身已经可用。若要在新机器上从零复现，按下面顺序执行
（下面的命令都在仓库根目录下运行）：

```powershell
# 1) 准备一个基础 Python 环境（本机用的是 Anaconda 的 pytorch 环境，Python 3.9）
#    已存在则跳过
D:\Anaconda\Scripts\conda.exe create -n pytorch python=3.9 -y

# 2) 在仓库根目录创建 venv，并用 --system-site-packages 复用基础环境里已装好的包
#    （这正是本仓库现有 .venv\pyvenv.cfg 里 home 指向 D:\Anaconda\envs\pytorch 的原因）
D:\Anaconda\envs\pytorch\python.exe -m venv --system-site-packages .venv

# 3) 让 pip 走国内镜像以加速（本机未使用，视网络情况可选）
#    清华源示例：
# .\.venv\Scripts\python.exe -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 把 micrograd 以可编辑模式装上（会生成 micrograd.egg-info\）
.\.venv\Scripts\python.exe -m pip install -e .

# 5) 安装 demo 与测试所需的依赖
.\.venv\Scripts\python.exe -m pip install numpy matplotlib scikit-learn pytest
.\.venv\Scripts\python.exe -m pip install nbconvert nbclient ipykernel jupyter
```

本次实测的校验命令（确认包都在、版本正确）：

```powershell
.\.venv\Scripts\python.exe -m pip show micrograd
.\.venv\Scripts\python.exe -c "import numpy, matplotlib, sklearn; print(numpy.__version__, matplotlib.__version__, sklearn.__version__)"
```

预期输出：`micrograd` 显示 `Editable project location` 指向仓库根目录，版本 `0.1.0`；
三个库版本分别为 `2.0.1`、`3.9.2`、`1.6.1`。

---

## 3. 测试命令与结果

```powershell
.\.venv\Scripts\python.exe -m pytest test/test_engine.py -v
```

实际输出（2026-10-05 当天两次运行，结果一致）：

```
platform win32 -- Python 3.9.25, pytest-8.4.2, pluggy-1.6.0
rootdir: ...\micrograd
plugins: anyio-4.12.1
collected 2 items

test/test_engine.py::test_sanity_check PASSED                            [ 50%]
test/test_engine.py::test_more_ops PASSED                                [100%]

============================== 2 passed in 5.05s ==============================
```

| 项目 | 结果 |
| --- | --- |
| 收集到的用例 | 2 |
| 通过 | 2（`test_sanity_check`、`test_more_ops`） |
| 失败 / 错误 | 0 |
| 耗时 | 约 4.6–5.1 s |
| 退出码 | 0 |

`test_sanity_check` 用一个小表达式手工核对 `Value` 的前向值与反向梯度；
`test_more_ops` 覆盖 `+ - * /`、`**`、`relu` 以及 `exp/tanh` 等运算符的梯度正确性。
两者同时通过说明自动求导引擎的反向传播实现与预期一致。

---

## 4. demo 执行命令与结果

### 命令

用 nbconvert 真正执行 notebook（而不是只打开），结果写到新文件，保留原始文件不动。
**必须在仓库根目录执行**，因为 notebook 里 `from micrograd.engine import Value`
依赖当前目录能导入本地包：

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute demo.ipynb `
    --output demo.executed.ipynb `
    --ExecutePreprocessor.timeout=900 `
    --ExecutePreprocessor.kernel_name=python3
```

### 执行结果

- 8 个代码单元全部执行成功，执行序号 1–8 连续，**没有任何报错**。
- 总耗时约 171 s（主要花在第 7 单元：纯 Python 实现的 100 步 SGD 训练）。
- 生成文件：`demo.executed.ipynb`（约 95 KB，含内嵌输出与图片）。

### notebook 做了什么

这是 micrograd 的经典二分类演示：

1. 导入 `random`、`numpy`、`matplotlib`，开启 `%matplotlib inline`；
2. 从本地包导入 `Value`、`Neuron`、`Layer`、`MLP`；
3. 固定随机种子 `np.random.seed(1337)` / `random.seed(1337)`；
4. 用 `sklearn.datasets.make_moons(n_samples=100, noise=0.1)` 造两月牙数据集，
   标签转成 ±1，并画散点图；
5. 初始化两层 MLP `MLP(2, [16, 16, 1])`，打印结构与参数数量；
6. 定义 SVM max-margin 损失（hinge loss + L2 正则，`alpha = 1e-4`），同时计算准确率；
7. 优化循环：100 步 SGD，学习率线性衰减 `learning_rate = 1.0 - 0.9*k/100`，
   每步 forward → `model.zero_grad()` → `total_loss.backward()` → 更新 `p.data`；
8. 在 `h = 0.25` 的网格上跑模型，画决策边界 + 数据点。

### 输出内容

第 5 单元（模型结构）：

```
MLP of [Layer of [ReLUNeuron(2) × 16], Layer of [ReLUNeuron(16) × 16], Layer of [LinearNeuron(16)]]
number of parameters 337
```

第 6 单元（初始状态）：

```
Value(data=0.8958441028683222, grad=0) 0.5
```

第 7 单元（训练过程，节选）：

| step | loss | accuracy |
| --- | --- | --- |
| 0 | 0.8958 | 50.0% |
| 1 | 1.7236 | 81.0% |
| 22 | 0.1173 | 97.0% |
| 40 | 0.0602 | 100.0%（首次） |
| 99 | 0.01098 | 100.0% |

第 8 单元（坐标范围）：

```
(-1.548639298268643, 1.951360701731357)
```

结论：训练后 loss 从 0.896 收敛到约 0.01098，准确率达到 100%。
后期 loss 基本不再下降，因为数据损失已经归零，剩下的几乎全是 L2 正则项。
之后用同样的随机种子重跑应能得到一致结果（种子在单元 3 就固定了）。

### 生成的图片

notebook 内生成了 **2 张图**，以 base64 PNG 形式内嵌在 `demo.executed.ipynb` 中；
另外已解码导出为独立文件，方便直接查看：

| 文件 | 来源单元 | 内容 | 大小 |
| --- | --- | --- | --- |
| `demo_executed_outputs/fig1_cell4.png` | cell 4 | make_moons 数据集散点图（jet 配色） | 27,512 B |
| `demo_executed_outputs/fig2_cell8.png` | cell 8 | 训练后的决策边界 + 数据点（Spectral 配色） | 26,465 B |

导出图片用的命令（可重复执行）：

```powershell
.\.venv\Scripts\python.exe -c "
import json, base64, os
nb = json.load(open('demo.executed.ipynb', encoding='utf-8'))
os.makedirs('demo_executed_outputs', exist_ok=True)
n = 0
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        if 'image/png' in o.get('data', {}):
            n += 1
            fn = 'demo_executed_outputs/fig%d_cell%d.png' % (n, i)
            open(fn, 'wb').write(base64.b64decode(o['data']['image/png']))
            print(fn)
"
```

---

## 5. 遇到的问题与解决

### 5.1 直接敲 `python` 提示“Python was not found”

- 现象：在 PowerShell 里执行 `python --version` 弹出
  “Python was not found; run without arguments to install from the Microsoft Store”，
  命令返回非 0。
- 原因：Windows 的“应用执行别名”把 `python.exe` 重定向到了 Microsoft Store，
  且真实解释器没有加进 PATH。
- 解决：不用裸 `python`，全程显式调用 `.\.venv\Scripts\python.exe`。
  仓库里已有的 `pytest_run.log` 也记录了同样的写法。

### 5.2 notebook 必须从仓库根目录执行

- 现象：`demo.ipynb` 里有 `from micrograd.engine import Value`。
  如果在别的目录执行，会直接 `ModuleNotFoundError: No module named 'micrograd'`，
  哪怕已经 `pip install -e .`。
- 原因：本仓库源码目录本身就叫 `micrograd`，从仓库根目录运行时
  Python 会优先导入当前目录下的同名包。
- 解决：始终 `cd` 到仓库根目录后再执行 nbconvert / pytest。

### 5.3 依赖是否需要安装的判断

- 做法：先跑一遍导入检查，确认 `numpy / matplotlib / sklearn / nbconvert /
  nbclient / ipykernel / jupyter_client` 是否可用，而不是无脑安装。
- 结果：全部已存在，**无需安装**，省掉了一次可能的版本冲突风险。
- 说明：因为 `.venv\pyvenv.cfg` 里 `include-system-site-packages = true`，
  它直接复用了 `D:\Anaconda\envs\pytorch` 里已有的科学计算与 Jupyter 组件。

### 5.4 zmq 在 Windows 上的 Proactor 事件循环警告

- 现象：nbconvert 启动内核时输出
  `RuntimeWarning: Proactor event loop does not implement add_reader ...`。
- 影响：无害的警告，与 notebook 内容和结果无关，内核照常启动并顺利完成执行。
- 处理：忽略。若要消除，可在执行前设置
  `asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())`。

### 5.5 中文路径导致的乱码

- 现象：控制台与 `pytest_run.log` 里仓库路径中的“学习”显示为 `ѧϰ`。
- 原因：Windows 控制台默认代码页（GBK/CP936）与 UTF-8 不一致，
  日志重定向时把中文字节按错误编码解码了。
- 影响：仅是显示问题，文件路径本身正确，pytest 与 notebook 均正常执行、正常写盘。
- 解决：日志写入时显式指定 UTF-8（例如 `Out-File -Encoding utf8`），
  或在执行前 `chcp 65001`。

### 5.6 执行耗时较长属正常现象

- 现象：`nbconvert --execute` 跑了约 171 s，中间长时间没有输出。
- 原因：第 7 单元是 100 步 SGD，每步都要对 100 个样本在前向和反向中创建大量
  `Value` 对象，micrograd 是纯 Python 标量实现，没有向量化，因此比 PyTorch 慢很多。
- 解决：不需要处理。给足超时即可（本次用 `--ExecutePreprocessor.timeout=900`）。

---

## 附：当天新增/生成的文件

以下内容为本次操作产生，当前处于未跟踪状态（`.gitignore` 只忽略了
`.ipynb_checkpoints/`）：

| 路径 | 说明 |
| --- | --- |
| `demo.executed.ipynb` | 已执行版本，含全部输出与两张内嵌图 |
| `demo_executed_outputs/fig1_cell4.png` | 数据集散点图 |
| `demo_executed_outputs/fig2_cell8.png` | 决策边界图 |
| `reproduction_notes.md` | 本文档 |

原始 `demo.ipynb` 未被修改。
