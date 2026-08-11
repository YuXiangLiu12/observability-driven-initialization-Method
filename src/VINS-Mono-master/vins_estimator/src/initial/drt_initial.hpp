#pragma once

#include <map>
#include <Eigen/Dense>

#include "drt_optimation.hpp"
#include "initial_alignment.h"

void gyroBiasEstimator(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &Bg);

// 文档模型外层迭代版：联合估计 bg / Rbc / td（drt_Exrot_dt.md §6）
// 每轮外层迭代：用当前 bg 重积分(repropagate) -> 多跨度帧对链式累积 -> 内层 LM 求增量
//              -> 更新状态 -> 增量/代价收敛则提前停止（最多 max_outer_iter 轮）
// min_pair_step：帧对最小跨度。帧离太近时帧间旋转量小，加噪声后 SNR 低，应跳过。
// robust_loss_scale：>0 时用 CauchyLoss(scale)（真实数据有外点，尺度应≈2~3×残差噪声地板，
//                   如 0.05）；=0 时纯最小二乘（仿真全内点，最优）。
void exrotEstimatorDocIterative(std::map<double, ImageFrame> &all_image_frame,
                                Eigen::Vector3d &Bg,
                                bool estimate_td = true,
                                bool estimate_Rbc = true,
                                int max_outer_iter = 5,
                                int max_pair_step = 4,
                                int min_pair_step = 2,
                                int min_features = 8,
                                double robust_loss_scale = 0.0,
                                double *td_out = nullptr);   // 输出优化后的 td（后检测需要）


// ============================================================================
// 可观测性门控（rotation_observability_analysis.md §5/§7-§11）
// ----------------------------------------------------------------------------
// 与 hessianConditionAnalysis（原始 H=JᵀJ + CRLB）的区别：本函数按文档做
//   1) §5 无量纲缩放 A_s = J·S，S = diag(ε_b I3, ε_R I3, ε_t)（ε 为应用误差容差）
//   2) §7 有效局部曲率 H_{q|n} = (P_n⊥ A_q)ᵀ(P_n⊥ A_q)：SVD 投影消去干扰量列空间
//   3) §8 无量纲敏感度半径 ρ = 1/√λ（td 标量 κ；Rbc/bg 各 3 维特征值）
//   4) §9 曲率独立率 η（区分"激励不足"与"参数强耦合"）
//   5) §10 无量纲曲率门控 + 数据量/陀螺饱和等工程检查
// 输出：OPTIMIZE_NOW（预检测通过，可运行 exrotEstimatorDocIterative）或
//       EXTEND_WINDOW（激励不足/耦合强，延长窗口）。§11 的 N_max 拒绝逻辑由调用方处理。
// 注意：线性化点 (bg, Rbc, td) 必须与后续优化器的初值一致；本函数会修改
//       all_image_frame 内预积分状态（repropagate），与迭代版一致。
// ============================================================================

// §11 预检测输出：是否进行优化 / 窗口延长
enum class ObsDecision
{
    OPTIMIZE_NOW,   // 预检测通过 → 运行联合旋转优化
    EXTEND_WINDOW   // 激励不足/参数耦合强 → 延长窗口（等待更多帧）
};

// 门控参数（§10）：无量纲尺度 ε（构造 S 用）+ 敏感度半径阈值 + 工程检查阈值
// ε 为用户设定的应用误差容差（2026-08-11）：td 5ms、Rbc 5°、bg 0.02 rad/s。
//   A_s = J·S，S = diag(ε_b I3, ε_R I3, ε_t)；σ_可达 = ρ·σ̂_noise·ε。
// ρ_max 标定（§10）：要求 σ_可达 ≤ ε ⟹ ρ ≤ 1/σ̂_noise。
//   实测收敛残差噪声地板 σ̂_noise≈0.02，取安全系数 c=0.5 → ρ_max = 0.5/0.02 = 25。
//   验证（10s 窗口，新 ε 下 ρ=(td 16.7, Rbc 11.1, bg 5.7)）全部 < 25 通过，
//   且实际恢复误差 (3.3ms, 0.38°, 0.0017) 均在新容差内。
struct ObsGateParams
{
    double eps_t = 5e-3;                                  // s    （td 容差 5ms）
    double eps_R = 5.0 * 3.14159265358979323846 / 180.0;  // rad  （Rbc 容差 5°）
    double eps_b = 2e-2;                                  // rad/s（bg 容差 0.02 rad/s）
    double rho_t_max = 25.0;          // ρ_td 上限（=0.5/σ̂_noise，2026-08-11 标定）
    double rho_R_max = 25.0;          // max_k ρ_bc,k 上限（=0.5/σ̂_noise）
    double rho_b_max = 25.0;          // max_k ρ_bg,k 上限（=0.5/σ̂_noise）
    double eta0 = 1e-2;               // 最小曲率独立率（§10）
    int    min_pairs = 4;             // 最少关键帧对数（§10）
    int    min_inliers = 50;          // 最少共视内点特征总数（§10）
    double svd_rel_thresh = 1e-6;     // SVD 截断相对阈值（§12.2，固定并记录）
    double max_gyro_norm = 30.0;      // 端点角速度量程（rad/s，>0 启用；需按 IMU 量程配置）
    bool   gate_bg = true;            // bg 是否参与门控（§10 三组都门控；bg 通常最先达标）
};

// 诊断输出（§14）：秩、有效曲率、敏感度半径、曲率独立率等
struct ObsDiagnostics
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int n_pairs = 0, n_inliers = 0, n_param = 0, rank_A = 0;
    double kappa_td = 0.0;                                   // κ_{td|bg,bc}
    Eigen::Vector3d mu_bc = Eigen::Vector3d::Zero();         // H_{bc|bg,t} 特征值（升序）
    Eigen::Vector3d nu_bg = Eigen::Vector3d::Zero();         // H_{bg|bc,t} 特征值（升序）
    double rho_td = 0.0;
    Eigen::Vector3d rho_bc = Eigen::Vector3d::Zero();        // 1/√μ_k（§8.2）
    Eigen::Vector3d rho_bg = Eigen::Vector3d::Zero();        // 1/√ν_k（§8.3）
    double eta_td = 0.0, eta_bc_min = 0.0, eta_bg_min = 0.0; // 曲率独立率（§9）
    double rms0 = 0.0;                                       // 基态残差 RMS
    bool   gyro_saturated = false;                           // 端点角速度饱和
    // ---- 后检测新增（§11 step 8-9）----
    double residual_mean = 0.0;    // 解处残差均值
    double residual_median = 0.0;  // 解处残差中位数
    double sigma_noise = 0.0;      // 噪声地板（稳健，= 解处残差中位数）
    bool   opt_healthy = true;     // 优化健康（mean < health_ratio × median）
    double rho_max_adaptive = 0.0; // 自适应阈值 clamp(c/σ̂_noise, rho_min, rho_max_bound)
};

// ============================================================================
// 可观测性后检测（§11 step 8-9）：优化后在解处重线性化复核，最终决定接受/拒绝
// ----------------------------------------------------------------------------
// 预检测在初值处做（噪声地板不可测），只能靠固定阈值兜底；优化收敛后残差到达
// 噪声地板 σ̂_noise，后检测据此用自适应阈值 ρ_max = c/σ̂_noise 做最终接受判据。
// 判定：
//   ACCEPT        ：解处 ρ ≤ ρ_max_adaptive 且 η ≥ η0 且优化健康 → 接受初始化
//   EXTEND_WINDOW ：优化健康（残差已到地板）但 ρ 仍不足 → 延长窗口重试，
//                   并更新 ρ_max_adaptive（σ̂ 可靠）
//   OPT_FAILED    ：优化不健康（残差未降到地板，局部极小/外点）→ σ̂ 不可靠，
//                   不更新阈值，延长窗口/换初值重试
// ============================================================================

// 后检测判定
enum class PostCheckVerdict
{
    ACCEPT,         // 接受旋转初始化
    EXTEND_WINDOW,  // 优化健康但窗口激励不足 → 更新阈值 + 延长窗口
    OPT_FAILED      // 优化失败（残差未到地板）→ 不更新阈值，延长窗口/换初值
};

// 后检测参数
struct PostCheckParams
{
    double safety_factor = 0.5;   // ρ_max = c/σ̂_noise 的安全系数
    double health_ratio  = 3.0;   // 健康度：mean < health_ratio × median 才算优化健康
    double rho_min       = 5.0;   // 自适应阈值下限（防 σ̂ 极小导致阈值过大）
    double rho_max_bound = 200.0; // 自适应阈值上限（防 σ̂ 极大导致阈值过小）
};

// 后检测：输入优化后的 (bg, Rbc, td)，返回判定 + 诊断（含 σ̂_noise 与自适应阈值）
PostCheckVerdict observabilityPostCheck(
    std::map<double, ImageFrame> &all_image_frame,
    const Eigen::Vector3d &Bg,               // 优化后 bg
    const Eigen::Quaterniond &q_bc,          // 优化后 Rbc
    double td,                               // 优化后 td
    bool estimate_td,
    bool estimate_Rbc,
    int max_pair_step,
    int min_pair_step,
    int min_features,
    double eps_lambda,
    const ObsGateParams &params,             // ε 容差 / svd_rel_thresh / η0
    const PostCheckParams &pc = PostCheckParams(),
    ObsDiagnostics *diag = nullptr);

// 可观测性门控（§11 预检测）：输入当前窗口数据 + 名义值 (bg, Rbc, td)，
// 输出 OPTIMIZE_NOW（可运行联合优化）或 EXTEND_WINDOW（延长窗口）。
ObsDecision observabilityGate(
    std::map<double, ImageFrame> &all_image_frame,
    const Eigen::Vector3d &Bg,               // 当前名义零偏（= 优化器初值）
    const Eigen::Quaterniond &q_bc,          // 当前名义外参 Rbc（= 优化器初值）
    double td,                               // 当前名义时间偏移（= 优化器初值）
    bool estimate_td = true,
    bool estimate_Rbc = true,
    int max_pair_step = 10,
    int min_pair_step = 4,
    int min_features = 50,
    double eps_lambda = 1e-20,
    const ObsGateParams &params = ObsGateParams(),
    ObsDiagnostics *diag = nullptr);

// ============================================================================
// 已知旋转下基于极线约束的帧间平移计算（fixed_rotation_camera_translation_initialization_corrected.md §4-§7）
// ----------------------------------------------------------------------------
// 输入：all_image_frame —— 每帧的相机旋转 frame.Rc0c（^c0 R_ci）已由旋转优化得到。
// 输出：写回 frame.Tc0c（= ^c0 p_ci，视觉尺度下的相机中心位置）、frame.R、frame.T。
// 返回：true = 成功写出位置；false = 数据不足 / 退化。
// 步骤（对应文档）：
//   1) 有序帧表 R[i]（第0帧为参考，必要时左乘 R0^T 归一化）            §3.2
//   2) 多跨度帧对收集共视 bearing，M_ij 谱诊断过滤                     §4.2/§6
//   3) 全局位置系统 A_p x_p = 0，H_p = A_p^T A_p 最小特征向量           §5.1
//   4) 用基线最大的帧对归一化视觉尺度                                   §5.2
//   5) 二视图正深度检查消解整体符号二义性                               §5.3
//   6) 写回 Rc0c / Tc0c / R / T                                        §7
// 注意：单目极线约束只恢复"视觉尺度"下的位置；米制尺度由后续
//       VisualIMUAlignment 恢复，本函数不触碰 TIC[0]（§8）。
//       阈值（tau_ratio / tau_energy）应按数据标定（§6），默认值较宽松。
bool translationEstimator(std::map<double, ImageFrame> &all_image_frame,
                          int min_features = 8,
                          int min_pair_step = 2,
                          int max_pair_step = 4,
                          double tau_ratio = 0.9,     // §6 谱比 λ1/λ2 上限
                          double tau_energy = 1e-8,   // §6 q_ij = λ2 能量下限
                          bool verbose = true);

// ============================================================================
// 消去速度后的尺度、重力与加速度计零偏闭式解算
// 文档：closed_form_scale_gravity_accel_bias_velocity_eliminated_规范化审核版.md
// ----------------------------------------------------------------------------
// 输入：all_image_frame —— frame.R（^c0 R_bi）、frame.T（视觉尺度相机中心）、
//       以及已用最终 bg 重积分（repropagate）的 pre_integration（δp/δv/J_b_a）。
// 输出：s = 单目视觉尺度；g_c0 = c0 系物理重力向量（向下，= -G_code）；
//       ba = 绝对加速度计零偏（= bar_ba + δb_a）。
// 返回：true = 解算成功（N≥5、rank=7、条件数≤tau_kappa、s>0）；false = 失败/退化。
// 方法（文档 §4-§7）：位置方程消去速度，构造固定 7 列线性系统 A_red x = b_red，
//       x = [s, g^c0, δb_a]；列主元 QR + SVD 最小二乘求解。
bool scaleGravityBiasEstimator(std::map<double, ImageFrame> &all_image_frame,
                               double &s,
                               Eigen::Vector3d &g_c0,
                               Eigen::Vector3d &ba,
                               double tau_gravity = 0.5, // ||g||-G 容差 (m/s²)，仅告警
                               double tau_kappa = 1e8,   // A_red 条件数上限（硬拒绝）
                               bool verbose = true);

// ============================================================================
// 视觉-惯性对齐：旋转优化结果 → 矫正预积分 → Rc0c → Tc0c → 尺度/重力/零偏
// 文档：fixed_rotation_camera_translation_initialization_corrected.md §3/§9
//       + closed_form_scale_gravity_accel_bias_velocity_eliminated_规范化审核版.md
// ----------------------------------------------------------------------------
// 输入：all_image_frame（含预积分与共视特征）；旋转优化估计的 bg / q_bc(Rbc) / td。
// 步骤：1) 用 bg 矫正 IMU 预积分（repropagate，§6.2）
//       2) 用优化后的 Rbc 逐帧累积相机旋转 Rc0c（§3.2/§3.3）
//       3) 调用 translationEstimator 计算相机平移 Tc0c（阈值更严格）
//       4) 调用 scaleGravityBiasEstimator 闭式解算尺度/物理重力/加计零偏
// 注意：当前工程预积分不含 td，按文档 §3.1/§3.3 此处置 td=0。
// 可选输出：s_out / g_c0_out / ba_out（scaleGravityBiasEstimator 的结果）。
void VisionIMuAlignment(std::map<double, ImageFrame> &all_image_frame,
                        Eigen::Vector3d bg,
                        Eigen::Quaterniond q_bc,
                        double td,
                        double *s_out = nullptr,
                        Eigen::Vector3d *g_c0_out = nullptr,
                        Eigen::Vector3d *ba_out = nullptr);
