"""完整静态像差仿真的统一配置。

依次运行本目录下的 step1 到 step5 脚本。正式实验时，步骤三可以由
真实相机采集替代，只要继续输出 SLM_rawN.mat 中的 imsdata 变量即可。
"""

from pathlib import Path

from workflows.static_simulation.workflow import SimulationSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


SETTINGS = SimulationSettings(
    # ---------- 输入与运行名称 ----------
    # 输入可以是普通二维/RGB 图片，也可以是三维光声 TIFF。
    # auto 会把普通图片按原流程读取；三维 TIFF 先减 2048、负值置零、沿第 0 维投影。
    project_root=PROJECT_ROOT,
    input_image=PROJECT_ROOT / "data" / "test.tif",
    # 开始新实验时，建议将 data_dir 最后一段和 scene_name 同时改成新名称。
    data_dir=PROJECT_ROOT / "data" / "test_static_zernike_50",
    result_root=PROJECT_ROOT,
    scene_name="test_static_zernike_50",
    input_mode="auto",  # 可选 auto、image、photoacoustic-volume
    photoacoustic_baseline=2048.0,
    photoacoustic_projection_axis=0,  # 按旧代码固定为 0；不要求恰好 512 层
    # 真实采集时填写存放 50 个三维 TIFF 的目录，并运行真实数据版步骤三。
    raw_measurement_dir=PROJECT_ROOT / "data" / "raw_photoacoustic_measurements",

    # ---------- 图像与孔径 ----------
    # size 必须是正偶数；None 表示整个方形区域都有光。
    size=256,
    aperture_height=None,
    num_frames=50,

    # ---------- 固定系统像差真值 ----------
    # 只在步骤一抽样一次；后续所有步骤读取同一份已保存的像差。
    # Noll 1–3 是 Piston/Tip/Tilt，本次从 4–15 中抽取中等强度混合像差。
    system_noll_start=4,
    system_noll_end=15,
    system_sigma=0.6,
    system_seed=20260730,

    # ---------- 已知 SLM 随机调制 ----------
    # 每帧用 Noll 1–15 的随机加权和产生一张新的 SLM 相位。
    slm_num_modes=15,
    slm_sigma=5.0,
    slm_seed=20260731,
    # -1 对应项目和论文数据约定 exp(-1j * proj_sim)，一般不要修改。
    phase_sign=-1,

    # ---------- 相机噪声与前向仿真 ----------
    # 第一轮闭环建议保持 0；以后可以增加固定种子的高斯噪声。
    noise_std=0.0,
    noise_seed=20260732,
    simulation_batch_size=8,
    generation_device="cuda",

    # ---------- NeuWS 重建训练 ----------
    # 当前 RTX 5070 Ti 上 256×256、50 帧、1000 轮约需 4–5 分钟。
    training_device="cuda",
    training_epochs=1000,
    training_batch_size=8,
    phase_layers=4,
    initial_learning_rate=1e-3,
    final_learning_rate=1e-3,
    # 每隔多少次训练迭代保存一次过程图；0 表示关闭过程图。
    visualization_frequency=1000,

    # ---------- 输出保护 ----------
    # False 会在同名结果已存在时停止，防止误覆盖。新实验优先改运行名称。
    overwrite=False,
)
