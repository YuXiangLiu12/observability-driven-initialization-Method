#include "drt_initial.hpp"
#include "../utility/tic_toc.h"   // TicToc 计时

#include <string>
#include <algorithm>
#include <iomanip>   // std::setw / std::left / std::setprecision（参数可观性表格）
#include <cmath>     // std::sqrt / std::max

// 旋转误差的轴角表示（用于打印）。
// 避免 Eigen eulerAngles(2,1,0) 在 gimbal-lock 附近（误差≈180° 或 ≈0° 时）的 ±180°
// 换分支伪影——实测 0.38° 的误差会被打印成 ypr≈(179.7,-179.9,179.7)。
static double rotationErrorDeg(const Eigen::Matrix3d &Rerr, Eigen::Vector3d *axis = nullptr)
{
    const double c = std::max(-1.0, std::min(1.0, (Rerr.trace() - 1.0) / 2.0));
    const double ang = std::acos(c) * 180.0 / M_PI;
    if (axis)
    {
        if (ang > 1e-6)
        {
            *axis = Eigen::Vector3d(Rerr(2, 1) - Rerr(1, 2),
                                    Rerr(0, 2) - Rerr(2, 0),
                                    Rerr(1, 0) - Rerr(0, 1)).normalized();
        }
        else
        {
            axis->setZero();
        }
    }
    return ang;
}

void gyroBiasEstimator(std::map<double, ImageFrame> &all_image_frame, Eigen::Vector3d &Bg)
{
    Eigen::Matrix3d Rbc_ = RIC[0];
    Eigen::Vector3d pbc_ = TIC[0];
    double td_ = 0.0;
    Eigen::Quaterniond q_bc_opt(Rbc_);

    // 专用参数块数组：避免 Eigen 栈对象 .data() 相邻导致 Ceres 参数块内存重叠(RegionsAlias)
    double para_bg[3] = {Bg(0), Bg(1), Bg(2)};
    double para_q[4]  = {q_bc_opt.x(), q_bc_opt.y(), q_bc_opt.z(), q_bc_opt.w()};
    double para_td[1] = {td_};

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1e-5);

    const int step = 1;

    auto frame_begin = all_image_frame.begin();

    for (auto it = frame_begin; it != all_image_frame.end(); ++it)
    {
        // 找到 step 步之后的帧
        auto next_it = it;
        for (int s = 0; s < step; ++s)
        {
            ++next_it;
            if (next_it == all_image_frame.end()) break;
        }
        if (next_it == all_image_frame.end()) break;
        
        auto &frame = it->second;
        auto &frame_next = next_it->second;

        // 累积中间帧的预积分旋转
        Eigen::Quaterniond q_total = Eigen::Quaterniond::Identity();
        Eigen::Matrix3d jacobian_total = Eigen::Matrix3d::Zero();

        auto mid_it = it;
        auto mid_next = std::next(mid_it);
        for (; mid_next != std::next(next_it); ++mid_it, ++mid_next)
        {
            if (mid_next == all_image_frame.end()) break;
            auto &mid_frame = mid_next->second;
            if (!mid_frame.pre_integration) continue;

            // 正确累积 Jacobian：链式法则
            // q_total_new = q_total * delta_q
            // d(q_total_new)/d(bg) = d(q_total)/d(bg) * delta_q + q_total * d(delta_q)/d(bg)
            // 近似：jacobian_total = jacobian_total + q_total * J_local (对于小旋转)
            Eigen::Matrix3d J_local = mid_frame.pre_integration->jacobian.block<3, 3>(O_R, O_BG);
            jacobian_total = jacobian_total + q_total.toRotationMatrix() * J_local;
            q_total = q_total * mid_frame.pre_integration->delta_q;
        }

        std::vector<Eigen::Vector3d> fis;
        std::vector<Eigen::Vector3d> fjs;
        std::vector<Eigen::Vector2d> vis; 
        std::vector<Eigen::Vector2d> vjs;
        
        for(const auto &feature_pair : frame.points)
        {
            int feature_id = feature_pair.first;

            if(frame_next.points.find(feature_id) != frame_next.points.end())
            {
                //归一化矢量
                fis.push_back(frame.points[feature_id][0].second.head<3>());
                fjs.push_back(frame_next.points[feature_id][0].second.head<3>());
                //特征点速度
                vis.push_back(frame.points[feature_id][0].second.tail<2>());
                vjs.push_back(frame_next.points[feature_id][0].second.tail<2>());
            }
        }

        if (fis.empty()) continue;  // 跳过无共视特征的帧对

        ceres::CostFunction *eigensolver_cost_function = 
            BiasSolverCostFunctor::CreateAccumulated(fis, fjs, vis, vjs, jacobian_total, q_total);
        problem.AddResidualBlock(eigensolver_cost_function, loss_function, para_bg, para_q, para_td);

    }

    // 设置四元数的流形约束，确保优化过程中四元数保持单位长度
    ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
    problem.SetParameterization(para_q, q_parameterization);

    //设置参数块为常量，不对这些参数进行优化
    problem.SetParameterBlockConstant(para_td);
    problem.SetParameterBlockConstant(para_q);
    std::cout << " don't estimate td and rotation ex" << std::endl;

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-20;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-20;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    // 从专用参数块读回
    Bg       = Eigen::Vector3d(para_bg[0], para_bg[1], para_bg[2]);
    q_bc_opt = Eigen::Quaterniond(para_q[3], para_q[0], para_q[1], para_q[2]).normalized();
    td_      = para_td[0];

    // 将优化结果拷贝回 Eigen 变量
    RIC[0] = q_bc_opt.toRotationMatrix();

}

void exrotEstimatorDoc(std::map<double, ImageFrame> &all_image_frame,
                       Eigen::Vector3d &Bg,
                       bool estimate_td,
                       bool estimate_Rbc)
{
    std::cout << "\n========== 旋转优化 drt_Exrot_dt.md ==========" << std::endl;

    Eigen::Matrix3d Rbc_ = RIC[0];
    double td_ = 0.0;                     
    Eigen::Quaterniond q_bc_opt(Rbc_);
    Eigen::Quaterniond q_bc_init(Rbc_);

    double para_bg[3] = {Bg(0), Bg(1), Bg(2)};
    double para_q[4]  = {q_bc_opt.x(), q_bc_opt.y(), q_bc_opt.z(), q_bc_opt.w()};
    double para_td[1] = {td_};

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1e-5);  // 与现有模型一致，保证对比公平

    const int step = 1;  // 帧对间隔，与现有模型一致（后续 Step 3 可扩展多跨度帧对）

    // 先收集每帧对的文档模型所需数据（预积分旋转/偏置雅可比/边界角速度/视线）
    struct DocPairData {
        std::vector<Eigen::Vector3d> fis, fjs;   // 帧 i / 帧 j 的特征视线
        Eigen::Quaterniond q_body;               // 预积分相对旋转 delta_q
        Eigen::Matrix3d J_bg;                    // dq_dbg = jacobian.block<3,3>(O_R,O_BG)
        Eigen::Vector3d bg_lin;                  // 积分时使用的偏置 linearized_bg
        Eigen::Vector3d w_start_raw;             // 起点边界原始角速度 linearized_gyr（区间首样本）
        Eigen::Vector3d w_end_raw;               // 终点边界原始角速度 gyr_1（区间末样本）
    };
    std::vector<DocPairData> pairs;

    auto frame_begin = all_image_frame.begin();
    for (auto it = frame_begin; it != all_image_frame.end(); ++it)
    {
        auto next_it = it;
        for (int s = 0; s < step; ++s)
        {
            ++next_it;
            if (next_it == all_image_frame.end()) break;
        }
        if (next_it == all_image_frame.end()) break;

        auto &frame = it->second;
        auto &frame_next = next_it->second;
        if (!frame_next.pre_integration) continue;  // 缺少预积分则跳过

        DocPairData pd;
        for (const auto &feature_pair : frame.points)
        {
            int feature_id = feature_pair.first;
            if (frame_next.points.find(feature_id) != frame_next.points.end())
            {
                pd.fis.push_back(frame.points[feature_id][0].second.head<3>());
                pd.fjs.push_back(frame_next.points[feature_id][0].second.head<3>());
            }
        }
        if (pd.fis.empty()) continue;  // 跳过无共视特征的帧对

        IntegrationBase *preint = frame_next.pre_integration;  // 帧对 (i,j) 的 [t_i, t_j] 区间
        pd.q_body       = preint->delta_q;
        pd.J_bg         = preint->jacobian.block<3, 3>(O_R, O_BG);
        pd.bg_lin       = preint->linearized_bg;
        pd.w_start_raw  = preint->linearized_gyr;  // 区间起点边界角速度（首样本）
        pd.w_end_raw    = preint->gyr_1;           // 区间终点边界角速度（末样本）
        pairs.push_back(pd);

        problem.AddResidualBlock(
            DocRotationCostFunctor::Create(pd.fis, pd.fjs, pd.q_body, pd.J_bg,
                                           pd.bg_lin, pd.w_start_raw, pd.w_end_raw),
            loss_function, para_bg, para_q, para_td);
    }

    // 四元数流形约束（para_q 内存顺序 x,y,z,w，与 Eigen::Quaternion coeffs() 一致）
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    problem.SetParameterization(para_q, q_parameterization);

    // 默认固定 td 与旋转外参，先只估计零偏（用户要求先验证零偏正确性）
    if (!estimate_td)  problem.SetParameterBlockConstant(para_td);
    if (!estimate_Rbc) problem.SetParameterBlockConstant(para_q);

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.gradient_tolerance = 1e-20;
    options.function_tolerance = 1e-20;
    options.parameter_tolerance = 1e-20;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    // 从专用参数块读回优化结果
    Bg       = Eigen::Vector3d(para_bg[0], para_bg[1], para_bg[2]);
    q_bc_opt = Eigen::Quaterniond(para_q[3], para_q[0], para_q[1], para_q[2]).normalized();
    td_      = para_td[0];

    // ---- 对比汇总 ----
    Eigen::Matrix3d Rbc_true;
    Rbc_true << 0, 0, -1,
                -1, 0, 0,
                 0, 1, 0;
    Eigen::Matrix3d Rbc_doc = q_bc_opt.toRotationMatrix();
    Eigen::Matrix3d Rbc_error = Rbc_doc.transpose() * Rbc_true;
    Eigen::Vector3d rbc_err_axis;
    const double rbc_err_deg = rotationErrorDeg(Rbc_error, &rbc_err_axis);

    std::cout << "[文档模型] bg = " << Bg.transpose()
              << " | Rbc error = " << rbc_err_deg << " deg (轴 " << rbc_err_axis.transpose() << ")"
              << " | td = " << td_ << std::endl;
    std::cout << "[文档模型] 参数状态: 固定td=" << (!estimate_td ? "是" : "否")
              << " 固定Rbc=" << (!estimate_Rbc ? "是" : "否") << std::endl;
}

// ============================================================================
// 文档模型外层迭代版：联合估计 bg / Rbc / td
// 迭代策略见 drt_Exrot_dt.md §6（区别于单次最小二乘，每轮都会重对齐/重积分/重新线性化）：
//   for outer = 1..max_outer_iter:
//     1. 用当前 bg 对每个帧间预积分 repropagate（重新积分角速度）        §6.2
//     2. 按当前 td 做端点补偿（cost functor 内部，文档 eq.5/6/12）       §6.1
//     3. 收集多跨度帧对 step=1..max_pair_step，链式累积 q_total / J_bg  §4.2
//     4. 固定本轮对齐与权重，内层 LM 求解增量 (dbg, dtheta, dtd)         §6.4/6.5
//     5. 更新状态: bg<-bg+dbg, Rbc<-Rbc*Exp(dtheta), td<-td+dtd          §6.6
//     6. 增量与代价下降均小于阈值则提前停止                             §6.8
// §8 提示：外层迭代不创造可观性，多跨度帧对 + 多轴变角速度运动才是信息源。
//          函数开头会对初始状态做 H=J^T W J 条件数检查。
// ============================================================================
void exrotEstimatorDocIterative(std::map<double, ImageFrame> &all_image_frame,
                                Eigen::Vector3d &Bg,
                                bool estimate_td,
                                bool estimate_Rbc,
                                int max_outer_iter,
                                int max_pair_step,
                                int min_pair_step,
                                int min_features,
                                double robust_loss_scale,
                                double *td_out)
{
    const double eps_lambda = 1e-20;   // 与文档模型一致（噪声更小时避免 sqrt 导数发散）
    // 参数校验：最小跨度不小于 1，且不超过最大跨度
    if (min_pair_step < 1) min_pair_step = 1;
    if (min_pair_step > max_pair_step) min_pair_step = max_pair_step;
    std::cout << "\n========== 旋转优化(外层迭代版) drt_Exrot_dt.md §6 ==========" << std::endl;
    std::cout << "[跨度] min_pair_step=" << min_pair_step
              << " max_pair_step=" << max_pair_step
              << " (帧间旋转量小、噪声主导的过近帧对将被跳过)" << std::endl;

    // ---- 当前状态（外层迭代的线性化点）----
    Eigen::Vector3d bg = Bg;
    Eigen::Quaterniond q_bc(RIC[0]);
    double td = 0.0;

    // ---- 计时（性能分析）：累计各阶段耗时 ----
    double t_collect_ms = 0.0;   // 收集帧对（repropagate + 共视匹配 + 链式累积）
    double t_obs_ms     = 0.0;   // §8 可观性检查（中心差分 Jacobian）
    double t_build_ms   = 0.0;   // 构建 Ceres problem（AddResidualBlock）
    double t_solve_ms   = 0.0;   // 内层 LM 求解
    double t_eval_ms    = 0.0;   // 残差评估（收敛代价）
    TicToc tt_total;             // 整个函数总时长

    // ---- 帧对数据结构 ----
    struct PairData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::vector<Eigen::Vector3d> fis, fjs;  // 共视特征归一化视线（帧 i / 帧 j）
        Eigen::Quaterniond q_total;             // 链式累积预积分旋转 R_bi_bj（含当前 bg 的重积分）
        Eigen::Matrix3d J_bg;                   // 链式累积 d(delta_q)/d(bg)（右扰动，与 gyroBiasEstimator 一致）
        Eigen::Vector3d w_start_raw;            // 区间起点边界原始角速度（t_i 处首样本 linearized_gyr）
        Eigen::Vector3d w_end_raw;              // 区间终点边界原始角速度（t_j 处末样本 gyr_1）
    };
    typedef std::vector<PairData, Eigen::aligned_allocator<PairData> > PairList;

    // ---- 收集帧对：用当前 bg 重积分 + 多跨度链式累积（§6.2 + §4.2）----
    auto collect_pairs = [&](const Eigen::Vector3d &cur_bg, PairList &pairs) {
        pairs.clear();

        // §6.2 用当前 bg 重新积分所有帧间预积分（更新 delta_q 与 J_bg）
        for (auto &kv : all_image_frame)
            if (kv.second.pre_integration)
                kv.second.pre_integration->repropagate(Eigen::Vector3d::Zero(), cur_bg);

        // §4.2 收集共视数量满足阈值、跨度 min_pair_step..max_pair_step 的帧对
        // min_pair_step：帧离太近（帧间旋转小）时信号弱、噪声主导，跳过以获得更好 SNR
        for (auto it = all_image_frame.begin(); it != all_image_frame.end(); ++it)
        {
            for (int step = min_pair_step; step <= max_pair_step; ++step)
            {
                auto jt = it;
                bool reachable = true;
                for (int s = 0; s < step; ++s)
                {
                    ++jt;
                    if (jt == all_image_frame.end()) { reachable = false; break; }
                }
                if (!reachable) break;

                const auto &frame_i = it->second;
                const auto &frame_j = jt->second;

                // 共视特征（归一化视线）
                std::vector<Eigen::Vector3d> fis, fjs;
                for (const auto &fp : frame_i.points)
                {
                    auto fjn = frame_j.points.find(fp.first);
                    if (fjn != frame_j.points.end())
                    {
                        fis.push_back(fp.second[0].second.head<3>());
                        fjs.push_back(fjn->second[0].second.head<3>());
                    }
                }
                if ((int)fis.size() < min_features) continue;

                // 链式累积 [t_i, t_j] 的预积分：帧 i+1..j 的 pre_integration 依次相乘
                //   q_total 累积：q_total = q_total * delta_q
                //   J_bg 累积（右扰动链式法则）：J_total = J_total + R(q_total)*J_local
                Eigen::Quaterniond q_total = Eigen::Quaterniond::Identity();
                Eigen::Matrix3d J_total = Eigen::Matrix3d::Zero();
                auto mit = it;
                bool complete = true;
                while (mit != jt)
                {
                    auto mnext = std::next(mit);
                    IntegrationBase *p = mnext->second.pre_integration;
                    if (!p) { complete = false; break; }
                    Eigen::Matrix3d J_local = p->jacobian.block<3, 3>(O_R, O_BG);
                    J_total = J_total + q_total.toRotationMatrix() * J_local;
                    q_total = q_total * p->delta_q;
                    mit = mnext;
                }
                if (!complete) continue;

                PairData pd;
                pd.fis = std::move(fis);
                pd.fjs = std::move(fjs);
                pd.q_total = q_total;
                pd.J_bg = J_total;
                pd.w_start_raw = std::next(it)->second.pre_integration->linearized_gyr; // t_i 处
                pd.w_end_raw   = jt->second.pre_integration->gyr_1;                    // t_j 处
                pairs.push_back(std::move(pd));
            }
        }
    };

    // ---- 计算给定状态下的残差向量（每帧对 e = sqrt(max(λmin, ελ))）----
    // bg_lin: 预积分线性化点（= collect_pairs 时的 bg，必须固定）；bg_val: 要评估的 bg 参数值。
    // 两者必须分开：若都传同一值，functor 内部 dbg = bg_val - bg_lin ≡ 0，bg 扰动被抵消，
    // 导致可观性检查的 bg 列恒为零（曾出现 3 个精确 0 特征值的假象）。
    auto eval_residuals = [&](const PairList &pairs, const Eigen::Vector3d &bg_lin,
                              const Eigen::Vector3d &bg_val,
                              const Eigen::Quaterniond &c_q, double c_td,
                              std::vector<double> &res) {
        res.clear();
        res.reserve(pairs.size());
        for (const auto &pd : pairs)
        {
            DocRotationCostFunctor cf(pd.fis, pd.fjs, pd.q_total, pd.J_bg, bg_lin,
                                      pd.w_start_raw, pd.w_end_raw, eps_lambda);
            double pbg[3] = {bg_val(0), bg_val(1), bg_val(2)};
            double pq[4]  = {c_q.x(), c_q.y(), c_q.z(), c_q.w()};
            double ptd[1] = {c_td};
            double r[1];
            cf(pbg, pq, ptd, r);
            res.push_back(r[0]);
        }
    };

    // ---- §8 可观性检查：初始状态处中心差分 Jacobian -> H = J^T J 特征值/条件数 ----
    // 仅在初始状态（λmin 明显 > ελ）下计算，避免 ελ 平台导致梯度为零的假象。
    // 注意：若初始状态已接近最优（如 Rbc 已为真值），该参数列梯度趋零属正常，
    //       此时 H 特征值偏小不代表运动激励不足（见下方 rms 判断）。
    auto observability_check = [&](const PairList &pairs) {
        // 估计参数维度：0..2 = bg, 3..5 = theta(Rbc 右扰动), 6 = td
        std::vector<int> est_dims = {0, 1, 2};
        if (estimate_Rbc) { est_dims.push_back(3); est_dims.push_back(4); est_dims.push_back(5); }
        if (estimate_td)  est_dims.push_back(6);

        std::vector<double> r0, rp, rm;
        eval_residuals(pairs, bg, bg, q_bc, td, r0);   // r0: 当前状态（bg_lin = bg_val = bg）
        const int n_res = (int)r0.size();
        double rms0 = 0.0;
        for (double r : r0) rms0 += r * r;
        rms0 = std::sqrt(rms0 / std::max(n_res, 1));
        const double h = 1e-6;
        Eigen::MatrixXd J(n_res, (int)est_dims.size());
        for (size_t k = 0; k < est_dims.size(); ++k)
        {
            int d = est_dims[k];
            Eigen::Vector3d bgp = bg, bgm = bg;
            Eigen::Quaterniond qp = q_bc, qm = q_bc;
            double tdp = td, tdm = td;
            if (d < 3)
            {
                bgp(d) += h; bgm(d) -= h;          // 只扰动 bg_val，bg_lin 保持 bg
            }
            else if (d < 6)
            {
                Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
                if (d == 4) axis = Eigen::Vector3d::UnitY();
                if (d == 5) axis = Eigen::Vector3d::UnitZ();
                qp = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(h, axis));
                qm = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(-h, axis));
            }
            else
            {
                tdp = td + h; tdm = td - h;
            }
            if (d < 3)
            {
                eval_residuals(pairs, bg, bgp, q_bc, td, rp);
                eval_residuals(pairs, bg, bgm, q_bc, td, rm);
            }
            else
            {
                eval_residuals(pairs, bg, bg, qp, tdp, rp);
                eval_residuals(pairs, bg, bg, qm, tdm, rm);
            }
            for (int i = 0; i < n_res; ++i)
                J(i, (int)k) = (rp[i] - rm[i]) / (2.0 * h);
        }
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
        Eigen::VectorXd evals = es.eigenvalues();
        double lam_max = evals(evals.size() - 1);
        double cond = lam_max / std::max(evals(0), 1e-30);
        std::cout << "[§8 可观性] 初始残差 rms=" << rms0
                  << " | H=J^T J 特征值: " << evals.transpose()
                  << " | 条件数 = " << cond << std::endl;
        // 仅在初始残差显著（状态未接近最优）时才判定运动激励是否不足
        if (rms0 > 1e-8 && evals(0) < 1e-12 * lam_max)
            std::cout << "[§8 警告] 存在弱可观方向（小特征值），运动激励或帧对集合不足，"
                      << "建议延长窗口 / 增加多轴旋转 / 固定部分参数" << std::endl;
        else if (rms0 <= 1e-8)
            std::cout << "[§8 提示] 初始状态已接近最优，梯度趋零，条件数仅供参考" << std::endl;
    };

    PairList pairs;
    double cost_prev = 0.0;
    int outer = 0;
    int iterations_done = 0;   // 实际执行的外层迭代次数（避免循环自然结束时 outer+1 多报 1）
    bool converged = false;
    for (outer = 0; outer < max_outer_iter; ++outer)
    {
        iterations_done = outer + 1;
        TicToc tt_collect;
        collect_pairs(bg, pairs);
        t_collect_ms += tt_collect.toc();
        if (pairs.empty())
        {
            std::cout << "[迭代] 无有效帧对，终止" << std::endl;
            break;
        }
        if (outer == 0)
        {
            TicToc tt_obs;
            observability_check(pairs);
            t_obs_ms += tt_obs.toc();
        }
        std::cout << "[外层 " << outer + 1 << "/" << max_outer_iter
                  << "] 帧对数 = " << pairs.size() << std::endl;

        // 专用参数块数组（避免 Eigen 栈对象 .data() 相邻导致 Ceres RegionsAlias 崩溃）
        double para_bg[3] = {bg(0), bg(1), bg(2)};
        double para_q[4]  = {q_bc.x(), q_bc.y(), q_bc.z(), q_bc.w()};
        double para_td[1] = {td};

        TicToc tt_build;
        ceres::Problem problem;
        // §5 权重：当前为统一权重（第一版约定），未实现逐帧对协方差权重 σ_λij²。
        // 损失函数：robust_loss_scale>0 时用 CauchyLoss(scale)。尺度必须匹配残差噪声地板
        // （真实数据 e~0.02，尺度应 ~0.05；若用 1e-5 会把梯度压到 ~1e-6 导致优化冻结）；
        // =0 时纯最小二乘（仿真全内点，最优）。
        ceres::LossFunction *loss = nullptr;
        if (robust_loss_scale > 0.0)
            loss = new ceres::CauchyLoss(robust_loss_scale);
        for (const auto &pd : pairs)
        {
            problem.AddResidualBlock(
                DocRotationCostFunctor::Create(pd.fis, pd.fjs, pd.q_total, pd.J_bg, bg,
                                               pd.w_start_raw, pd.w_end_raw, eps_lambda),
                loss, para_bg, para_q, para_td);
        }

        ceres::LocalParameterization *q_manifold = new ceres::EigenQuaternionParameterization();
        problem.SetParameterization(para_q, q_manifold);
        if (!estimate_td)  problem.SetParameterBlockConstant(para_td);
        if (!estimate_Rbc) problem.SetParameterBlockConstant(para_q);
        t_build_ms += tt_build.toc();

        // §6.5 内层 LM
        ceres::Solver::Options options;
        options.max_num_iterations = 50;
        options.function_tolerance  = 1e-14;
        options.gradient_tolerance  = 1e-12;
        options.parameter_tolerance = 1e-14;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = false;
        TicToc tt_solve;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        double solve_ms = tt_solve.toc();
        t_solve_ms += solve_ms;

        // ---- 读回增量 ----
        Eigen::Vector3d bg_new(para_bg[0], para_bg[1], para_bg[2]);
        Eigen::Quaterniond q_new(para_q[3], para_q[0], para_q[1], para_q[2]);
        q_new.normalize();
        double td_new = para_td[0];

        Eigen::Vector3d dbg = bg_new - bg;
        double dtd = td_new - td;
        // 右扰动：q_new = q_bc * Exp(dtheta)  =>  dq = q_bc^-1 * q_new（与文档 §1.3 一致）
        Eigen::Quaterniond dq = q_bc.inverse() * q_new;
        Eigen::AngleAxisd aa(dq);
        Eigen::Vector3d dtheta = aa.angle() * aa.axis();

        double cost_new = 0.0;
        {
            TicToc tt_eval;
            std::vector<double> rv;
            // 收敛代价：bg_lin 用本轮线性化点 bg（collect_pairs 时的），bg_val 用求解后的 bg_new
            eval_residuals(pairs, bg, bg_new, q_new, td_new, rv);
            for (double r : rv) cost_new += r * r;
            t_eval_ms += tt_eval.toc();
        }

        std::cout << "  内层: iters=" << summary.iterations.size()
                  << " 代价=" << cost_new
                  << " | 求解=" << solve_ms << "ms"
                  << " | dbg=" << dbg.transpose()
                  << " | dtheta(deg)=" << (dtheta * 180.0 / M_PI).transpose()
                  << " | dtd=" << dtd << std::endl;

        // ---- §6.6 更新状态 ----
        bg = bg_new;
        q_bc = q_new;
        td = td_new;

        // ---- §6.8 收敛判断：增量足够小，且代价不再下降 ----
        double rel_cost_change = (outer == 0) ? 1.0
            : std::fabs(cost_prev - cost_new) / std::max(cost_prev, 1e-30);
        bool tiny_delta = (dbg.norm() < 1e-9 && dtheta.norm() < 1e-9 && std::fabs(dtd) < 1e-9);
        if (tiny_delta || (outer > 0 && rel_cost_change < 1e-8 && cost_new <= cost_prev))
        {
            converged = true;
            std::cout << "[外层] " << (tiny_delta ? "增量趋于零" : "代价收敛")
                      << "，提前停止" << std::endl;
            break;
        }
        cost_prev = cost_new;
    }

    // ---- 收敛后残差分布（实测内点噪声地板，用于校准鲁棒核尺度）----
    {
        std::vector<double> rv;
        eval_residuals(pairs, bg, bg, q_bc, td, rv);   // 最终状态下每帧对 e = sqrt(λmin)
        if (!rv.empty())
        {
            std::vector<double> r = rv;
            std::sort(r.begin(), r.end());
            auto pct = [&](double p) { return r[(size_t)(p * (r.size() - 1))]; };
            double mean = 0.0;
            for (double x : rv) mean += x;
            mean /= (double)rv.size();
            std::cout << "[残差分布] 帧对数=" << rv.size()
                      << " 均值=" << mean
                      << " 中位数=" << pct(0.5)
                      << " p90=" << pct(0.9)
                      << " max=" << r.back() << std::endl;
            // CauchyLoss 尺度取 2~3×中位数：内点权重≥0.8，外点被压制
            std::cout << "[提示] CauchyLoss 尺度建议: c ≈ " << 2.0 * pct(0.5)
                      << " ~ " << 3.0 * pct(0.5) << std::endl;
        }
    }

    // ---- 计时汇总 ----
    double t_total_ms = tt_total.toc();
    std::cout << "[计时] 总耗时=" << t_total_ms << "ms | 收集帧对=" << t_collect_ms
              << "ms | §8检查=" << t_obs_ms << "ms | 建Problem=" << t_build_ms
              << "ms | 内层求解=" << t_solve_ms << "ms | 残差评估=" << t_eval_ms << "ms" << std::endl;

    // ---- 结果对比（真值仅用于仿真验证）----
    Eigen::Matrix3d Rbc_true;
    // Rbc_true << 0, 0, -1,
    //             -1, 0, 0,
    //              0, 1, 0;

    Rbc_true << 0.0148655429818, -0.999880929698, 0.00414029679422,
                0.999557249008, 0.0149672133247, 0.025715529948,
                -0.0257744366974, 0.00375618835797, 0.999660727178;

    Eigen::Matrix3d Rbc_doc = q_bc.toRotationMatrix();
    Eigen::Matrix3d Rbc_error = Rbc_doc.transpose() * Rbc_true;
    Eigen::Vector3d rbc_err_axis;
    const double rbc_err_deg = rotationErrorDeg(Rbc_error, &rbc_err_axis);

    std::cout << "[迭代版] bg = " << bg.transpose()
              << " | Rbc error = " << rbc_err_deg << " deg (轴 " << rbc_err_axis.transpose() << ")"
              << " | td = " << td
              << " | 外层迭代 = " << iterations_done
              << " | 收敛 = " << (converged ? "是" : "否") << std::endl;

    // ---- 写回 ----
    Bg = bg;
    RIC[0] = q_bc.toRotationMatrix();
    if (td_out) *td_out = td;   // 输出优化后的 td（后检测需要）
}

// ============================================================================
// 残差模型海森矩阵分析（参考 exrotEstimatorDocIterative 内的 §8 可观性检查）
// 在给定状态 (bg, Rbc=RIC[0], td=0) 下：
//   1. 用当前 bg 对帧间预积分 repropagate（保证 bg_lin == bg_val，与 collect_pairs 一致）
//   2. 收集多跨度帧对并链式累积 q_total / J_bg（同 exrotEstimatorDocIterative）
//   3. 对残差 e = sqrt(max(λmin(M), ελ)) 做中心差分，得 Jacobian J (n_res × n_param)
//   4. 计算 Gauss-Newton 海森 H = J^T J，特征值分解 + 条件数 κ = λ_max/λ_min
//   5. 输出参数可观性：协方差 C=σ̂²·H⁻¹、每参数 1σ 不确定度 σ_i=σ̂·√C_ii、
//      可观性得分 SNR_i=tol_i/σ_i、可观性排序、最弱/最强方向 1σ δ=σ̂/√λ。
// 参数维度：0..2 = bg，3..5 = Rbc 右扰动 θ，6 = td（按 estimate_* 开关裁剪）
// 容差（"稳定可观"的判据：σ_i < tol_i 即 SNR_i > 1）：
//   tol_bg    : bg 允许的 1σ 不确定度（rad/s），默认 1e-3
//   tol_theta : Rbc 允许的 1σ 不确定度（rad），默认 1° = π/180
//   tol_td    : td 允许的 1σ 不确定度（s），默认 2e-3
// 返回值：true = 旋转外参(Rbc)与时间延时(td)均稳定可观；false = Rbc 或 td 存在未达容差。
//         （bg 不参与触发门控——它通常最先达标，且残差签名随跨度增长、信息最强。）
// 注意：本函数会修改 all_image_frame 内预积分状态（repropagate），与迭代版一致。
//       若当前状态已接近最优（λmin < ελ 平台），对应参数列梯度趋零属正常现象。
// ============================================================================
bool hessianConditionAnalysis(std::map<double, ImageFrame> &all_image_frame,
                              const Eigen::Vector3d &Bg,
                              bool estimate_td,
                              bool estimate_Rbc,
                              int max_pair_step,
                              int min_pair_step,
                              int min_features,
                              double eps_lambda,
                              double tol_bg,
                              double tol_theta,
                              double tol_td)
{

    std::cout << "all_image_frame frame num " << all_image_frame.size() << std::endl;

    if (min_pair_step < 1) min_pair_step = 1;
    if (min_pair_step > max_pair_step) min_pair_step = max_pair_step;

    // ---- 计时（性能分析）：累计各阶段耗时 ----
    double t_reprop_ms  = 0.0;   // 重积分（repropagate）
    double t_collect_ms = 0.0;   // 收集帧对（共视匹配 + 链式累积）
    double t_jac_ms     = 0.0;   // 中心差分 Jacobian（残差评估）
    double t_hess_ms    = 0.0;   // H=J^T J + 特征值分解
    TicToc tt_total;             // 整个函数总时长

    std::cout << "\n========== 海森矩阵条件数/特征值分析 (残差模型 e=sqrt(max(λmin,ελ))) ==========" << std::endl;
    std::cout << "[跨度] min_pair_step=" << min_pair_step
              << " max_pair_step=" << max_pair_step
              << " | 估计参数: bg" << (estimate_Rbc ? "+Rbc" : "")
              << (estimate_td ? "+td" : "") << std::endl;

    // ---- 当前线性化点（与 exrotEstimatorDocIterative 初始化一致）----
    Eigen::Vector3d bg = Bg;
    Eigen::Quaterniond q_bc(RIC[0]);
    double td = 0.0;

    // ---- 1. 用当前 bg 重积分（bg_lin = bg_val，避免 functor 内 dbg 被抵消）----
    {
        TicToc tt;
        for (auto &kv : all_image_frame)
            if (kv.second.pre_integration)
                kv.second.pre_integration->repropagate(Eigen::Vector3d::Zero(), bg);
        t_reprop_ms += tt.toc();
    }

    // ---- 2. 收集帧对（多跨度链式累积 q_total / J_bg）----
    struct PairData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::vector<Eigen::Vector3d> fis, fjs;  // 共视特征归一化视线（帧 i / 帧 j）
        Eigen::Quaterniond q_total;             // 链式累积预积分旋转 R_bi_bj
        Eigen::Matrix3d J_bg;                   // 链式累积 d(delta_q)/d(bg)（右扰动）
        Eigen::Vector3d w_start_raw;            // 区间起点边界原始角速度（linearized_gyr）
        Eigen::Vector3d w_end_raw;              // 区间终点边界原始角速度（gyr_1）
    };
    typedef std::vector<PairData, Eigen::aligned_allocator<PairData> > PairList;
    PairList pairs;

    TicToc tt_collect;
    for (auto it = all_image_frame.begin(); it != all_image_frame.end(); ++it)
    {
        for (int step = min_pair_step; step <= max_pair_step; ++step)
        {
            auto jt = it;
            bool reachable = true;
            for (int s = 0; s < step; ++s)
            {
                ++jt;
                if (jt == all_image_frame.end()) { reachable = false; break; }
            }
            if (!reachable) break;

            const auto &frame_i = it->second;
            const auto &frame_j = jt->second;

            // 共视特征（归一化视线）
            std::vector<Eigen::Vector3d> fis, fjs;
            for (const auto &fp : frame_i.points)
            {
                auto fjn = frame_j.points.find(fp.first);
                if (fjn != frame_j.points.end())
                {
                    fis.push_back(fp.second[0].second.head<3>());
                    fjs.push_back(fjn->second[0].second.head<3>());
                }
            }
            if ((int)fis.size() < min_features) continue;

            // 链式累积 [t_i, t_j] 的预积分：q_total = q_total * delta_q；
            // J_bg 右扰动链式法则：J_total = J_total + R(q_total)*J_local
            Eigen::Quaterniond q_total = Eigen::Quaterniond::Identity();
            Eigen::Matrix3d J_total = Eigen::Matrix3d::Zero();
            auto mit = it;
            bool complete = true;
            while (mit != jt)
            {
                auto mnext = std::next(mit);
                IntegrationBase *p = mnext->second.pre_integration;
                if (!p) { complete = false; break; }
                Eigen::Matrix3d J_local = p->jacobian.block<3, 3>(O_R, O_BG);
                J_total = J_total + q_total.toRotationMatrix() * J_local;
                q_total = q_total * p->delta_q;
                mit = mnext;
            }
            if (!complete) continue;

            PairData pd;
            pd.fis = std::move(fis);
            pd.fjs = std::move(fjs);
            pd.q_total = q_total;
            pd.J_bg = J_total;
            pd.w_start_raw = std::next(it)->second.pre_integration->linearized_gyr; // t_i 处
            pd.w_end_raw   = jt->second.pre_integration->gyr_1;                    // t_j 处
            pairs.push_back(std::move(pd));
        }
    }
    t_collect_ms += tt_collect.toc();

    if (pairs.empty())
    {
        std::cout << "[海森分析] 无有效帧对，无法计算" << std::endl;
        std::cout << "[海森分析·计时] 总耗时 = " << tt_total.toc() << "ms"
                  << " (重积分=" << t_reprop_ms << "ms 收集帧对=" << t_collect_ms << "ms)" << std::endl;
        return false;   // 无帧对 → 无法满足容差
    }
    std::cout << "[海森分析] 帧对数 = " << pairs.size() << std::endl;

    // ---- 残差评估：bg_lin 固定为 bg，bg_val 可变（与迭代版 eval_residuals 一致）----
    auto eval_residuals = [&](const PairList &pl, const Eigen::Vector3d &bg_lin,
                              const Eigen::Vector3d &bg_val,
                              const Eigen::Quaterniond &c_q, double c_td,
                              std::vector<double> &res) {
        res.clear();
        res.reserve(pl.size());
        for (const auto &pd : pl)
        {
            DocRotationCostFunctor cf(pd.fis, pd.fjs, pd.q_total, pd.J_bg, bg_lin,
                                      pd.w_start_raw, pd.w_end_raw, eps_lambda);
            double pbg[3] = {bg_val(0), bg_val(1), bg_val(2)};
            double pq[4]  = {c_q.x(), c_q.y(), c_q.z(), c_q.w()};
            double ptd[1] = {c_td};
            double r[1];
            cf(pbg, pq, ptd, r);
            res.push_back(r[0]);
        }
    };

    // ---- 参数维度：0..2 = bg, 3..5 = theta(Rbc 右扰动), 6 = td ----
    std::vector<int> est_dims = {0, 1, 2};
    if (estimate_Rbc) { est_dims.push_back(3); est_dims.push_back(4); est_dims.push_back(5); }
    if (estimate_td)  est_dims.push_back(6);

    std::vector<std::string> names;
    names.push_back("bg_x"); names.push_back("bg_y"); names.push_back("bg_z");
    if (estimate_Rbc) { names.push_back("θx"); names.push_back("θy"); names.push_back("θz"); }
    if (estimate_td)  names.push_back("td");
    const int n_param = (int)est_dims.size();
    const int n_res   = (int)pairs.size();

    // ---- 3. 中心差分 Jacobian ----
    std::vector<double> r0;
    eval_residuals(pairs, bg, bg, q_bc, td, r0);   // 基态残差（bg_lin = bg_val = bg）
    double rms0 = 0.0;
    for (double r : r0) rms0 += r * r;
    rms0 = std::sqrt(rms0 / std::max(n_res, 1));
    std::cout << "[海森分析] 基态残差 rms = " << rms0
              << " (ε_λ = " << eps_lambda << ")" << std::endl;

    const double h = 1e-6;
    TicToc tt_jac;
    Eigen::MatrixXd J(n_res, n_param);
    for (size_t k = 0; k < est_dims.size(); ++k)
    {
        int d = est_dims[k];
        std::vector<double> rp, rm;
        if (d < 3)
        {
            // 只扰动 bg_val，bg_lin 保持 bg
            Eigen::Vector3d bgp = bg, bgm = bg;
            bgp(d) += h; bgm(d) -= h;
            eval_residuals(pairs, bg, bgp, q_bc, td, rp);
            eval_residuals(pairs, bg, bgm, q_bc, td, rm);
        }
        else if (d < 6)
        {
            // Rbc 右扰动：q_bc * Exp(±h·axis)
            Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
            if (d == 4) axis = Eigen::Vector3d::UnitY();
            if (d == 5) axis = Eigen::Vector3d::UnitZ();
            Eigen::Quaterniond qp = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(h, axis));
            Eigen::Quaterniond qm = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(-h, axis));
            eval_residuals(pairs, bg, bg, qp, td, rp);
            eval_residuals(pairs, bg, bg, qm, td, rm);
        }
        else
        {
            eval_residuals(pairs, bg, bg, q_bc, td + h, rp);
            eval_residuals(pairs, bg, bg, q_bc, td - h, rm);
        }
        for (int i = 0; i < n_res; ++i)
            J(i, (int)k) = (rp[i] - rm[i]) / (2.0 * h);
    }
    t_jac_ms += tt_jac.toc();

    // ---- 4. Gauss-Newton 海森 H = J^T J：特征值分解 + 条件数 ----
    TicToc tt_hess;
    Eigen::MatrixXd H = J.transpose() * J;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    Eigen::VectorXd evals = es.eigenvalues();
    double lam_max = evals(evals.size() - 1);
    double lam_min = evals(0);
    double cond = lam_max / std::max(lam_min, 1e-30);
    t_hess_ms += tt_hess.toc();

    std::cout << "[海森分析] 参数维度 = " << n_param
              << " (bg:" << 3 << " Rbc:" << (estimate_Rbc ? 3 : 0)
              << " td:" << (estimate_td ? 1 : 0) << ")" << std::endl;
    std::cout << "[海森分析] H = J^T J 特征值 = " << evals.transpose() << std::endl;
    std::cout << "[海森分析] λ_max = " << lam_max << " | λ_min = " << lam_min
              << " | 条件数 κ = " << cond
              << " | log10(κ) = " << std::log10(cond) << std::endl;

    // 各参数对角元素（可分离可观性的粗略指标）
    // std::cout << "[海森分析] diag(H): ";
    // for (int k = 0; k < n_param; ++k)
    //     std::cout << names[k] << "=" << H(k, k) << "  ";
    // std::cout << std::endl;

    // 最弱 / 最强可观方向（特征向量）
    Eigen::VectorXd v_min = es.eigenvectors().col(0);
    Eigen::VectorXd v_max = es.eigenvectors().col(evals.size() - 1);
    std::cout << "[海森分析] 最弱可观方向 v_min = ";
    for (int k = 0; k < n_param; ++k) std::cout << names[k] << "=" << v_min(k) << "  ";
    std::cout << std::endl;
    std::cout << "[海森分析] 最强可观方向 v_max = ";
    for (int k = 0; k < n_param; ++k) std::cout << names[k] << "=" << v_max(k) << "  ";
    std::cout << std::endl;

    // ======================================================================
    // 参数可观性 / 不确定度分析
    // ----------------------------------------------------------------------
    // 理论公式（残差模型 e_i = sqrt(max(λmin_i, ελ)), i = 1..n_res）：
    //   J      = ∂e/∂p ∈ R^{n_res×n_param}       中心差分 Jacobian（h=1e-6）
    //   H      = JᵀJ                             Gauss-Newton 近似海森（曲率矩阵）
    //   σ̂²     = (1/n_res)·Σ_i e_i²               残差方差估计（= rms0²）
    //   C      = σ̂²·H⁻¹ = σ̂²·Σ_k (v_k v_kᵀ)/λ_k  参数协方差（CRLB 下界）
    //   σ_i    = σ̂·√(C_ii)                        参数 p_i 的 1σ 不确定度
    //   δ_k    = σ̂/√λ_k                           沿特征方向 v_k 的 1σ 不确定度
    //   SNR_i  = tol_i / σ_i                       可观性得分（越大越可观）
    //   判据   : 全部参数稳定可观 ⟺ min_i SNR_i > 1 ⟺ max_i σ_i < tol_i
    // ----------------------------------------------------------------------
    // "稳定可观的不确定度" = 容差 tol_i（σ_i 必须 < tol_i 才算稳定可观）：
    //   bg:    tol_bg    = 1e-3 rad/s   (1 mrad/s)
    //   Rbc:   tol_theta = 1°           (0.01745 rad)
    //   td:    tol_td    = 2e-3 s       (2 ms)
    // ======================================================================
    std::cout << "[海森分析·理论] e=sqrt(max(λmin,ελ)) | J=∂e/∂p | H=JᵀJ | "
              << "σ̂²=(1/n)·Σe_i² | C=σ̂²·H⁻¹=σ̂²·Σ_k(v_k v_kᵀ)/λ_k | "
              << "σ_i=σ̂·√C_ii | δ_k=σ̂/√λ_k | SNR_i=tol_i/σ_i" << std::endl;
    std::cout << "[海森分析] 噪声 σ̂ = " << rms0 << " (基态残差 rms)" << std::endl;

    // Hinv = H⁻¹ = Σ_k v_k v_kᵀ/λ_k（特征分解求和式，比直接求逆数值更稳）。
    // 注意：Hinv 只存 H⁻¹，不乘 σ̂²；σ_i = σ̂·√Hinv_ii 才是正确 CRLB。
    // 曾有的 bug：Hinv 又乘了 rms0²、且 sig 再乘 rms0 → σ_i 被额外缩放 rms0 倍，
    // 导致 σ_i 偏小 rms0 倍、SNR 虚高 1/rms0 倍（对 rms0≈0.19 即约 5 倍）。
    Eigen::MatrixXd Hinv = Eigen::MatrixXd::Zero(n_param, n_param);
    for (int k = 0; k < n_param; ++k)
        Hinv += es.eigenvectors().col(k) * es.eigenvectors().col(k).transpose()
                / std::max(evals(k), 1e-30);

    // 每参数 1σ 不确定度 σ_i = σ̂·√((H⁻¹)_ii)、容差 tol_i、可观性 SNR_i = tol_i/σ_i
    std::vector<double> sig(n_param, 0.0), tol(n_param, 1.0), snr(n_param, 0.0);
    for (int k = 0; k < n_param; ++k)
    {
        sig[k] = rms0 * std::sqrt(std::max(Hinv(k, k), 0.0));
        if      (names[k] == "td")        tol[k] = tol_td;      // 时间延迟
        else if (names[k][0] == 'b')      tol[k] = tol_bg;      // bg_x/bg_y/bg_z
        else                              tol[k] = tol_theta;   // θx/θy/θz
        snr[k] = tol[k] / std::max(sig[k], 1e-30);
    }

    std::cout << "[海森分析] 参数          1σ不确定度σ_i   容差tol_i      SNR=tol/σ   判据(σ<tol)" << std::endl;
    for (int k = 0; k < n_param; ++k)
    {
        std::cout << "[海森分析] " << std::left << std::setw(13) << names[k]
                  << "  " << std::setw(16) << sig[k]
                  << "  " << std::setw(13) << tol[k]
                  << "  " << std::setw(10) << snr[k]
                  << "  " << (snr[k] > 1.0 ? "✓ 稳定可观" : "✗ 未达容差") << std::endl;
    }
    std::cout << std::right;

    // 可观性排序（SNR 降序：越靠前越可观）
    std::vector<int> order(n_param);
    for (int k = 0; k < n_param; ++k) order[k] = k;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return snr[a] > snr[b]; });
    std::cout << "[海森分析] 可观性排序(SNR降序): ";
    for (size_t r = 0; r < order.size(); ++r)
    {
        int k = order[r];
        std::cout << names[k] << "(SNR=" << snr[k] << ")";
        if (r + 1 < order.size()) std::cout << " > ";
    }
    std::cout << std::endl;

    // 方向不确定度 δ = σ̂/√λ（沿最弱 / 最强特征方向）
    double delta_min = rms0 / std::sqrt(std::max(lam_min, 1e-30));
    double delta_max = rms0 / std::sqrt(std::max(lam_max, 1e-30));
    std::cout << "[海森分析] 方向1σ: 最弱方向 δ_min=σ̂/√λ_min=" << delta_min
              << " | 最强方向 δ_max=σ̂/√λ_max=" << delta_max << std::endl;

    // 总判据：全部参数稳定可观 ⟺ 所有 SNR_i > 1（即 σ_i < tol_i）
    bool all_observable = true;
    int worst = -1;
    double worst_snr = 1e30;
    for (int k = 0; k < n_param; ++k)
    {
        if (snr[k] < worst_snr) { worst_snr = snr[k]; worst = k; }
        if (snr[k] <= 1.0) all_observable = false;
    }
    if (all_observable)
    {
        std::cout << "[海森分析] 结论: 全部参数稳定可观 (min SNR=" << worst_snr
                  << " > 1)" << std::endl;
    }
    else
    {
        // 枚举所有未达容差（SNR≤1）的参数，并标出最差的一个
        std::cout << "[海森分析] 结论: 存在未达容差(SNR≤1)的参数: ";
        bool first_fail = true;
        for (int k = 0; k < n_param; ++k)
        {
            if (snr[k] <= 1.0)
            {
                if (!first_fail) std::cout << ", ";
                first_fail = false;
                std::cout << names[k] << "(SNR=" << snr[k] << ", σ=" << sig[k]
                          << " > tol=" << tol[k] << ")";
            }
        }
        std::cout << " | 最差: " << names[worst] << " (SNR=" << worst_snr << ")" << std::endl;
    }

    // 触发判据：只要求旋转外参(θx/θy/θz)和时间延时(td)稳定可观（bg 通常最先达标，
    // 且在本门控中不作为约束），全部满足 → true；否则 → false。
    bool trigger_ok = true;
    for (int k = 0; k < n_param; ++k)
    {
        if (names[k] == "bg_x" || names[k] == "bg_y" || names[k] == "bg_z")
            continue;   // bg 不参与触发判据
        if (snr[k] <= 1.0) trigger_ok = false;
    }
    std::cout << "[海森分析] 触发判据(Rbc+td 稳定可观) = "
              << (trigger_ok ? "满足 ✓" : "不满足 ✗") << std::endl;

    // 弱可观方向警告（仅当基态残差显著时才判定运动激励不足）
    if (rms0 > 1e-8 && lam_min < 1e-12 * lam_max)
    {
        std::cout << "[海森分析] 警告: 存在弱可观方向（λ_min << λ_max），运动激励或帧对集合不足，"
                  << "建议延长窗口 / 增加多轴旋转 / 固定部分参数" << std::endl;
    }
    else if (rms0 <= 1e-8)
    {
        std::cout << "[海森分析] 提示: 当前状态已接近最优（残差≈0，进入 ελ 平台），"
                  << "梯度趋零，条件数仅供参考" << std::endl;
    }

    // ---- 计时汇总 ----
    double t_total_ms = tt_total.toc();
    std::cout << "[海森分析·计时] 总耗时 = " << t_total_ms << "ms"
              << " | 重积分 = " << t_reprop_ms << "ms"
              << " | 收集帧对 = " << t_collect_ms << "ms"
              << " | 中心差分J = " << t_jac_ms << "ms"
              << " | H/特征分解 = " << t_hess_ms << "ms" << std::endl;

    // 触发判据：Rbc(θ) 与 td 均满足容差 → true；存在未达容差 → false（bg 不参与门控）
    return trigger_ok;
}

// ============================================================================
// 可观测性门控 + 自适应窗口预检测（rotation_observability_analysis.md §5/§7-§11）
// ----------------------------------------------------------------------------
// 与 hessianConditionAnalysis（原始 H=JᵀJ + CRLB）的区别：
//   原始 H 把"运动激励不足"和"参数强耦合"混为一谈（§9 警告）。本函数按文档做：
//     1) §5 无量纲缩放 A_s = J·S，S = diag(ε_b I3, ε_R I3, ε_t)
//     2) §7 对每个目标量 q ∈ {td, θbc, bg}，SVD 投影消去干扰量 n 的列空间：
//           A_{q|n}^{eff} = P_n⊥ A_q，H_{q|n} = (A_{q|n}^{eff})ᵀ(A_{q|n}^{eff})
//     3) §8 无量纲敏感度半径 ρ = 1/√λ（td 标量 κ；Rbc/bg 各 3 维特征值）
//     4) §9 曲率独立率 η（原始曲率大+独立率大=激励充分；原始大+η小=参数混淆）
//     5) §10 无量纲曲率门控 + 数据量/陀螺饱和等工程检查
//   输出 OPTIMIZE_NOW（可运行联合优化）或 EXTEND_WINDOW（延长窗口）。
// 参数维度：0..2 = bg，3..5 = θbc（Rbc 右扰动），6 = td（按 estimate_* 开关裁剪）
// ============================================================================

// §7 有效局部曲率：H_{q|n} = (P_n⊥ A_q)ᵀ(P_n⊥ A_q)
// A_s: 无量纲雅可比 (n_res × n_param)；q_cols/n_cols: 目标量/干扰量列索引
// P_n⊥ = I − Q_n Q_nᵀ，Q_n = A_n 截断 SVD 有效列空间基（§7/§12.2，不显式求逆）
static Eigen::MatrixXd effectiveCurvature(const Eigen::MatrixXd &A_s,
                                          const std::vector<int> &q_cols,
                                          const std::vector<int> &n_cols,
                                          double svd_rel_thresh,
                                          int *rank_An)
{
    const int n_res = (int)A_s.rows();
    Eigen::MatrixXd An(n_res, (int)n_cols.size());
    for (size_t c = 0; c < n_cols.size(); ++c)
        An.col((int)c) = A_s.col(n_cols[c]);
    Eigen::MatrixXd Aq(n_res, (int)q_cols.size());
    for (size_t c = 0; c < q_cols.size(); ++c)
        Aq.col((int)c) = A_s.col(q_cols[c]);

    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(n_res, n_res);
    int r = 0;
    if (An.cols() > 0)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(An, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd s = svd.singularValues();
        const double s_max = (s.size() > 0) ? s(0) : 0.0;
        for (int i = 0; i < (int)s.size(); ++i)
            if (s(i) > svd_rel_thresh * s_max) ++r;
        if (r > 0)
        {
            const Eigen::MatrixXd Un = svd.matrixU().leftCols(r);
            P = Eigen::MatrixXd::Identity(n_res, n_res) - Un * Un.transpose();
        }
    }
    if (rank_An) *rank_An = r;
    const Eigen::MatrixXd Aq_eff = P * Aq;
    return Aq_eff.transpose() * Aq_eff;
}

// §9 标量独立率：η_td = κ_{td|n} / (A_s,tᵀ A_s,t)，0 ≤ η ≤ 1；分母≈0 → 不可辨识 → 0
// 数值稳健性：分母（原始曲率）≤ 1e-12 时 td 列雅可比≈0（如恒角速度 → J_t≈0，
// 即不可观，§13.3），此时 κ/raw 是 0/0 不定式，直接判 η=0 而非打印噪声比值。
static double independenceRateScalar(double kappa, double raw_curvature)
{
    if (raw_curvature <= 1e-12) return 0.0;
    return std::max(0.0, std::min(1.0, kappa / raw_curvature));
}

// §9 广义最小特征值独立率（Rbc/bg 用）：
//   η_{q,min} = λ_min( (A_s,qᵀA_s,q)^{+1/2} H_{q|n} (A_s,qᵀA_s,q)^{+1/2} )
// 原始曲率 B=A_s,qᵀA_s,q 奇异 → 伪逆平方根（SVD 限定到非零奇异值子空间，§9/§12.2）；
// 有效曲率在该子空间仍有零特征值 → 对应方向不可辨识 → η=0（不报告无意义的 0/0）。
static double independenceRateMin(const Eigen::MatrixXd &Aq,
                                  const Eigen::MatrixXd &Hqn,
                                  double svd_rel_thresh)
{
    const int m = (int)Aq.cols();
    if (m == 0) return 0.0;
    const Eigen::MatrixXd B = Aq.transpose() * Aq;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esB(B);
    const Eigen::VectorXd lam = esB.eigenvalues();
    const double lam_max = lam(m - 1);
    Eigen::MatrixXd Bph = Eigen::MatrixXd::Zero(m, m);
    bool any = false;
    for (int k = 0; k < m; ++k)
    {
        if (lam(k) > svd_rel_thresh * lam_max)
        {
            Bph += esB.eigenvectors().col(k) * esB.eigenvectors().col(k).transpose()
                   / std::sqrt(lam(k));
            any = true;
        }
    }
    if (!any) return 0.0;   // 原始曲率全零 → 目标方向完全不可辨识
    const Eigen::MatrixXd G = Bph * Hqn * Bph;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esG(G);
    return std::max(0.0, esG.eigenvalues()(0));   // SelfAdjointEigenSolver 升序，取最小
}

ObsDecision observabilityGate(
    std::map<double, ImageFrame> &all_image_frame,
    const Eigen::Vector3d &Bg,
    const Eigen::Quaterniond &q_bc,
    double td,
    bool estimate_td,
    bool estimate_Rbc,
    int max_pair_step,
    int min_pair_step,
    int min_features,
    double eps_lambda,
    const ObsGateParams &params,
    ObsDiagnostics *diag)
{
    if (min_pair_step < 1) min_pair_step = 1;
    if (min_pair_step > max_pair_step) min_pair_step = max_pair_step;

    ObsDiagnostics d;
    std::cout << "\n========== 可观测性门控 (rotation_observability_analysis.md §5/§7-§11) ==========" << std::endl;
    std::cout << "[门控] 窗口帧数=" << all_image_frame.size()
              << " | 跨度 " << min_pair_step << ".." << max_pair_step
              << " | 估计参数: bg" << (estimate_Rbc ? "+Rbc" : "")
              << (estimate_td ? "+td" : "")
              << " | ελ=" << eps_lambda
              << " | S尺度 ε=(bg " << params.eps_b << ", Rbc " << params.eps_R
              << ", td " << params.eps_t << ")" << std::endl;

    // ---- 1. 用当前名义 bg 重积分（bg_lin = bg_val，避免 functor 内 dbg 被抵消）----
    for (auto &kv : all_image_frame)
        if (kv.second.pre_integration)
            kv.second.pre_integration->repropagate(Eigen::Vector3d::Zero(), Bg);

    // ---- 2. 收集多跨度帧对（链式累积 q_total / J_bg，同 exrotEstimatorDocIterative）----
    struct PairData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::vector<Eigen::Vector3d> fis, fjs;  // 共视特征归一化视线（帧 i / 帧 j）
        Eigen::Quaterniond q_total;             // 链式累积预积分旋转 R_bi_bj
        Eigen::Matrix3d J_bg;                   // 链式累积 d(delta_q)/d(bg)（右扰动）
        Eigen::Vector3d w_start_raw;            // 区间起点边界原始角速度（linearized_gyr）
        Eigen::Vector3d w_end_raw;              // 区间终点边界原始角速度（gyr_1）
    };
    typedef std::vector<PairData, Eigen::aligned_allocator<PairData> > PairList;
    PairList pairs;
    int n_inliers = 0;
    for (auto it = all_image_frame.begin(); it != all_image_frame.end(); ++it)
    {
        for (int step = min_pair_step; step <= max_pair_step; ++step)
        {
            auto jt = it;
            bool reachable = true;
            for (int s = 0; s < step; ++s)
            {
                ++jt;
                if (jt == all_image_frame.end()) { reachable = false; break; }
            }
            if (!reachable) break;
            const auto &frame_i = it->second;
            const auto &frame_j = jt->second;
            std::vector<Eigen::Vector3d> fis, fjs;
            for (const auto &fp : frame_i.points)
            {
                auto fjn = frame_j.points.find(fp.first);
                if (fjn != frame_j.points.end())
                {
                    fis.push_back(fp.second[0].second.head<3>());
                    fjs.push_back(fjn->second[0].second.head<3>());
                }
            }
            if ((int)fis.size() < min_features) continue;
            Eigen::Quaterniond q_total = Eigen::Quaterniond::Identity();
            Eigen::Matrix3d J_total = Eigen::Matrix3d::Zero();
            auto mit = it;
            bool complete = true;
            while (mit != jt)
            {
                auto mnext = std::next(mit);
                IntegrationBase *p = mnext->second.pre_integration;
                if (!p) { complete = false; break; }
                Eigen::Matrix3d J_local = p->jacobian.block<3, 3>(O_R, O_BG);
                J_total = J_total + q_total.toRotationMatrix() * J_local;
                q_total = q_total * p->delta_q;
                mit = mnext;
            }
            if (!complete) continue;
            PairData pd;
            pd.fis = std::move(fis);
            pd.fjs = std::move(fjs);
            pd.q_total = q_total;
            pd.J_bg = J_total;
            pd.w_start_raw = std::next(it)->second.pre_integration->linearized_gyr; // t_i 处
            pd.w_end_raw   = jt->second.pre_integration->gyr_1;                    // t_j 处
            n_inliers += (int)pd.fis.size();
            pairs.push_back(std::move(pd));
        }
    }
    d.n_pairs = (int)pairs.size();
    d.n_inliers = n_inliers;

    // ---- 数据量检查（§10）：帧对数 / 内点总数不足 → 直接延长窗口 ----
    if (d.n_pairs < params.min_pairs || d.n_inliers < params.min_inliers)
    {
        std::cout << "[门控] 数据量不足: 帧对=" << d.n_pairs << "(需≥" << params.min_pairs
                  << ") 内点=" << d.n_inliers << "(需≥" << params.min_inliers
                  << ") → 延长窗口" << std::endl;
        if (diag) *diag = d;
        return ObsDecision::EXTEND_WINDOW;
    }

    // ---- 3. 残差评估 + 中心差分 Jacobian（bg 只扰动 bg_val，bg_lin 固定）----
    auto eval_residuals = [&](const PairList &pl, const Eigen::Vector3d &bg_lin,
                              const Eigen::Vector3d &bg_val,
                              const Eigen::Quaterniond &c_q, double c_td,
                              std::vector<double> &res) {
        res.clear();
        res.reserve(pl.size());
        for (const auto &pd : pl)
        {
            DocRotationCostFunctor cf(pd.fis, pd.fjs, pd.q_total, pd.J_bg, bg_lin,
                                      pd.w_start_raw, pd.w_end_raw, eps_lambda);
            double pbg[3] = {bg_val(0), bg_val(1), bg_val(2)};
            double pq[4]  = {c_q.x(), c_q.y(), c_q.z(), c_q.w()};
            double ptd[1] = {c_td};
            double r[1];
            cf(pbg, pq, ptd, r);
            res.push_back(r[0]);
        }
    };

    std::vector<int> est_dims = {0, 1, 2};
    std::vector<int> bg_cols  = {0, 1, 2};
    std::vector<int> rbc_cols;
    std::vector<int> td_col;
    if (estimate_Rbc)
    {
        rbc_cols = {3, 4, 5};
        est_dims.insert(est_dims.end(), rbc_cols.begin(), rbc_cols.end());
    }
    if (estimate_td)
    {
        td_col = {6};
        est_dims.push_back(6);
    }
    const int n_param = (int)est_dims.size();
    const int n_res   = d.n_pairs;
    d.n_param = n_param;

    std::vector<double> r0;
    eval_residuals(pairs, Bg, Bg, q_bc, td, r0);
    double rms0 = 0.0;
    for (double r : r0) rms0 += r * r;
    rms0 = std::sqrt(rms0 / std::max(n_res, 1));
    d.rms0 = rms0;
    std::cout << "[门控] 帧对=" << d.n_pairs << " 内点=" << d.n_inliers
              << " | 基态残差 rms=" << rms0 << std::endl;

    // 状态已接近最优（λmin < ελ 平台）：残差梯度趋零，曲率分析无意义（§3.2）。
    // 数据量已达标且残差≈0 → 当前名义值已拟合数据，优化只是确认性收敛。
    if (rms0 <= 10.0 * std::sqrt(std::max(eps_lambda, 1e-30)))
    {
        std::cout << "[门控] 基态残差接近 ελ 平台（状态已接近最优），预检测通过（确认性优化）" << std::endl;
        if (diag) *diag = d;
        return ObsDecision::OPTIMIZE_NOW;
    }

    // 中心差分 Jacobian J (n_res × n_param)，步长 h=1e-6
    const double h = 1e-6;
    Eigen::MatrixXd J(n_res, n_param);
    for (size_t k = 0; k < est_dims.size(); ++k)
    {
        const int dd = est_dims[k];
        std::vector<double> rp, rm;
        if (dd < 3)
        {
            Eigen::Vector3d bgp = Bg, bgm = Bg;
            bgp(dd) += h; bgm(dd) -= h;
            eval_residuals(pairs, Bg, bgp, q_bc, td, rp);
            eval_residuals(pairs, Bg, bgm, q_bc, td, rm);
        }
        else if (dd < 6)
        {
            // Rbc 右扰动（§1.1）：q_bc * Exp(±h·axis)，与优化器更新方向一致
            Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
            if (dd == 4) axis = Eigen::Vector3d::UnitY();
            if (dd == 5) axis = Eigen::Vector3d::UnitZ();
            const Eigen::Quaterniond qp = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(h, axis));
            const Eigen::Quaterniond qm = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(-h, axis));
            eval_residuals(pairs, Bg, Bg, qp, td, rp);
            eval_residuals(pairs, Bg, Bg, qm, td, rm);
        }
        else
        {
            eval_residuals(pairs, Bg, Bg, q_bc, td + h, rp);
            eval_residuals(pairs, Bg, Bg, q_bc, td - h, rm);
        }
        for (int i = 0; i < n_res; ++i)
            J(i, (int)k) = (rp[i] - rm[i]) / (2.0 * h);
    }

    // ---- 4. §5 无量纲缩放 A_s = J·S，S = diag(ε_b I3, ε_R I3, ε_t) ----
    Eigen::MatrixXd A_s = J;
    for (size_t k = 0; k < est_dims.size(); ++k)
    {
        const int dd = est_dims[k];
        const double s = (dd < 3) ? params.eps_b : ((dd < 6) ? params.eps_R : params.eps_t);
        A_s.col((int)k) *= s;
    }
    // 无量纲总曲率 H_s = A_sᵀA_s（§5/§14 汇总，供与 hessianConditionAnalysis 对照）
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esHs(A_s.transpose() * A_s);
    const Eigen::VectorXd evals_s = esHs.eigenvalues();
    const double lam_s_max = evals_s(evals_s.size() - 1);
    const double cond_s = lam_s_max / std::max(evals_s(0), 1e-30);
    std::cout << "[门控] H_s=A_sᵀA_s 特征值=" << evals_s.transpose()
              << " | 条件数=" << cond_s << std::endl;
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_s, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd s = svd.singularValues();
        const double s_max = (s.size() > 0) ? s(0) : 0.0;
        for (int i = 0; i < (int)s.size(); ++i)
            if (s(i) > params.svd_rel_thresh * s_max) ++d.rank_A;
    }
    std::cout << "[门控] rank(A_s) = " << d.rank_A << " / " << n_param << std::endl;

    // ---- 5. §7/§8 有效局部曲率 + §8 敏感度半径 + §9 曲率独立率 ----
    // td: q = t_d，n = [bg, θbc]（§8.1）
    if (estimate_td)
    {
        std::vector<int> ncols = bg_cols;
        ncols.insert(ncols.end(), rbc_cols.begin(), rbc_cols.end());
        const Eigen::MatrixXd Htd = effectiveCurvature(A_s, td_col, ncols,
                                                       params.svd_rel_thresh, nullptr);
        d.kappa_td = Htd(0, 0);
        const double raw_td = A_s.col(td_col[0]).squaredNorm();
        d.eta_td = independenceRateScalar(d.kappa_td, raw_td);
        d.rho_td = 1.0 / std::sqrt(std::max(d.kappa_td, 1e-30));
        std::cout << "[门控] td: κ_{td|bg,bc}=" << d.kappa_td
                  << " | 原始曲率 A_s,tᵀA_s,t=" << raw_td
                  << " | ρ_td=" << d.rho_td << " | η_td=" << d.eta_td << std::endl;
    }
    // Rbc: q = θbc，n = [bg, t_d]（§8.2）
    if (estimate_Rbc)
    {
        std::vector<int> ncols = bg_cols;
        ncols.insert(ncols.end(), td_col.begin(), td_col.end());
        const Eigen::MatrixXd Hbc = effectiveCurvature(A_s, rbc_cols, ncols,
                                                       params.svd_rel_thresh, nullptr);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hbc);
        d.mu_bc = es.eigenvalues();                     // 升序
        for (int k = 0; k < 3; ++k)
            d.rho_bc(k) = 1.0 / std::sqrt(std::max(d.mu_bc(k), 1e-30));
        Eigen::MatrixXd Aqbc(n_res, 3);
        for (int c = 0; c < 3; ++c) Aqbc.col(c) = A_s.col(rbc_cols[c]);
        d.eta_bc_min = independenceRateMin(Aqbc, Hbc, params.svd_rel_thresh);
        std::cout << "[门控] Rbc: μ=" << d.mu_bc.transpose()
                  << " | ρ_bc=" << d.rho_bc.transpose()
                  << " | η_bc,min=" << d.eta_bc_min << std::endl;
    }
    // bg: q = b_g，n = [θbc, t_d]（§8.3）
    {
        std::vector<int> ncols = rbc_cols;
        ncols.insert(ncols.end(), td_col.begin(), td_col.end());
        const Eigen::MatrixXd Hbg = effectiveCurvature(A_s, bg_cols, ncols,
                                                       params.svd_rel_thresh, nullptr);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hbg);
        d.nu_bg = es.eigenvalues();                     // 升序
        for (int k = 0; k < 3; ++k)
            d.rho_bg(k) = 1.0 / std::sqrt(std::max(d.nu_bg(k), 1e-30));
        Eigen::MatrixXd Aqbg(n_res, 3);
        for (int c = 0; c < 3; ++c) Aqbg.col(c) = A_s.col(bg_cols[c]);
        d.eta_bg_min = independenceRateMin(Aqbg, Hbg, params.svd_rel_thresh);
        std::cout << "[门控] bg: ν=" << d.nu_bg.transpose()
                  << " | ρ_bg=" << d.rho_bg.transpose()
                  << " | η_bg,min=" << d.eta_bg_min << std::endl;
    }

    // ---- 6. §10 陀螺饱和检查：端点角速度（td 端点补偿用）是否超量程 ----
    if (params.max_gyro_norm > 0.0)
    {
        for (const auto &pd : pairs)
        {
            if (pd.w_start_raw.norm() > params.max_gyro_norm ||
                pd.w_end_raw.norm() > params.max_gyro_norm)
            {
                d.gyro_saturated = true;
                break;
            }
        }
    }
    if (d.gyro_saturated)
        std::cout << "[门控] 警告: 端点角速度超出量程(>" << params.max_gyro_norm
                  << " rad/s)，td 端点补偿不可靠 → 延长窗口" << std::endl;

    // ---- 7. §10 无量纲曲率门控判据 ----
    bool pass = true;
    if (estimate_td  && d.rho_td > params.rho_t_max) pass = false;
    if (estimate_Rbc && d.rho_bc.maxCoeff() > params.rho_R_max) pass = false;
    if (params.gate_bg && d.rho_bg.maxCoeff() > params.rho_b_max) pass = false;
    if (estimate_td  && d.eta_td < params.eta0) pass = false;
    if (estimate_Rbc && d.eta_bc_min < params.eta0) pass = false;
    if (params.gate_bg && d.eta_bg_min < params.eta0) pass = false;
    if (d.gyro_saturated) pass = false;

    std::cout << "[门控] 判据: ρ_td=" << d.rho_td << "(≤" << params.rho_t_max << ")"
              << " | maxρ_bc=" << (estimate_Rbc ? d.rho_bc.maxCoeff() : 0.0)
              << "(≤" << params.rho_R_max << ")"
              << " | maxρ_bg=" << d.rho_bg.maxCoeff() << "(≤" << params.rho_b_max << ")"
              << " | η_td=" << d.eta_td << " η_bc,min=" << d.eta_bc_min
              << " η_bg,min=" << d.eta_bg_min << "(≥" << params.eta0 << ")"
              << " | 陀螺饱和=" << (d.gyro_saturated ? "是" : "否") << std::endl;
    std::cout << "[门控] 结论: " << (pass ? "预检测通过 → 运行联合优化 (OPTIMIZE_NOW)"
                                          : "激励不足/耦合强 → 延长窗口 (EXTEND_WINDOW)")
              << std::endl;

    if (diag) *diag = d;
    return pass ? ObsDecision::OPTIMIZE_NOW : ObsDecision::EXTEND_WINDOW;
}

// ============================================================================
// 可观测性后检测（rotation_observability_analysis.md §11 step 8-9）
// ----------------------------------------------------------------------------
// 预检测在初值处评估（残差被系统误差主导 → 噪声地板测不到），只能靠固定阈值兜底；
// 优化收敛后残差到达噪声地板 σ̂_noise，后检测在解处重线性化并：
//   1) 测 σ̂_noise = 解处残差中位数（稳健）
//   2) 健康度检查：mean < health_ratio×median 才算优化健康（区分"窗口不足"与"优化失败"）
//   3) 计算自适应阈值 ρ_max = clamp(c/σ̂_noise, rho_min, rho_max_bound)
//   4) 解处 ρ ≤ ρ_max 且 η ≥ η0 → ACCEPT；否则 EXTEND_WINDOW（σ̂ 可靠 → 更新阈值）
//   5) 不健康 → OPT_FAILED（σ̂ 虚高不可靠 → 不更新阈值，延长窗口/换初值）
// 参数维度：0..2 = bg，3..5 = θbc（Rbc 右扰动），6 = td（按 estimate_* 开关裁剪）
// 注意：本函数会修改 all_image_frame 内预积分状态（repropagate），与预检测一致。
// ============================================================================
PostCheckVerdict observabilityPostCheck(
    std::map<double, ImageFrame> &all_image_frame,
    const Eigen::Vector3d &Bg,
    const Eigen::Quaterniond &q_bc,
    double td,
    bool estimate_td,
    bool estimate_Rbc,
    int max_pair_step,
    int min_pair_step,
    int min_features,
    double eps_lambda,
    const ObsGateParams &params,
    const PostCheckParams &pc,
    ObsDiagnostics *diag)
{
    if (min_pair_step < 1) min_pair_step = 1;
    if (min_pair_step > max_pair_step) min_pair_step = max_pair_step;

    ObsDiagnostics d;
    std::cout << "\n========== 可观测性后检测 (rotation_observability_analysis.md §11 step 8-9) ==========" << std::endl;
    std::cout << "[后检测] 窗口帧数=" << all_image_frame.size()
              << " | 解处 bg=" << Bg.transpose()
              << " | td=" << td
              << " | ε=(bg " << params.eps_b << ", Rbc " << params.eps_R
              << ", td " << params.eps_t << ")" << std::endl;

    // ---- 1. 用优化后 bg 重积分（bg_lin = bg_val = Bg，解处线性化点）----
    for (auto &kv : all_image_frame)
        if (kv.second.pre_integration)
            kv.second.pre_integration->repropagate(Eigen::Vector3d::Zero(), Bg);

    // ---- 2. 收集多跨度帧对（链式累积 q_total / J_bg，同预检测）----
    struct PairData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::vector<Eigen::Vector3d> fis, fjs;  // 共视特征归一化视线（帧 i / 帧 j）
        Eigen::Quaterniond q_total;             // 链式累积预积分旋转 R_bi_bj
        Eigen::Matrix3d J_bg;                   // 链式累积 d(delta_q)/d(bg)（右扰动）
        Eigen::Vector3d w_start_raw;            // 区间起点边界原始角速度（linearized_gyr）
        Eigen::Vector3d w_end_raw;              // 区间终点边界原始角速度（gyr_1）
    };
    typedef std::vector<PairData, Eigen::aligned_allocator<PairData> > PairList;
    PairList pairs;
    int n_inliers = 0;
    for (auto it = all_image_frame.begin(); it != all_image_frame.end(); ++it)
    {
        for (int step = min_pair_step; step <= max_pair_step; ++step)
        {
            auto jt = it;
            bool reachable = true;
            for (int s = 0; s < step; ++s)
            {
                ++jt;
                if (jt == all_image_frame.end()) { reachable = false; break; }
            }
            if (!reachable) break;
            const auto &frame_i = it->second;
            const auto &frame_j = jt->second;
            std::vector<Eigen::Vector3d> fis, fjs;
            for (const auto &fp : frame_i.points)
            {
                auto fjn = frame_j.points.find(fp.first);
                if (fjn != frame_j.points.end())
                {
                    fis.push_back(fp.second[0].second.head<3>());
                    fjs.push_back(fjn->second[0].second.head<3>());
                }
            }
            if ((int)fis.size() < min_features) continue;
            Eigen::Quaterniond q_total = Eigen::Quaterniond::Identity();
            Eigen::Matrix3d J_total = Eigen::Matrix3d::Zero();
            auto mit = it;
            bool complete = true;
            while (mit != jt)
            {
                auto mnext = std::next(mit);
                IntegrationBase *p = mnext->second.pre_integration;
                if (!p) { complete = false; break; }
                Eigen::Matrix3d J_local = p->jacobian.block<3, 3>(O_R, O_BG);
                J_total = J_total + q_total.toRotationMatrix() * J_local;
                q_total = q_total * p->delta_q;
                mit = mnext;
            }
            if (!complete) continue;
            PairData pd;
            pd.fis = std::move(fis);
            pd.fjs = std::move(fjs);
            pd.q_total = q_total;
            pd.J_bg = J_total;
            pd.w_start_raw = std::next(it)->second.pre_integration->linearized_gyr; // t_i 处
            pd.w_end_raw   = jt->second.pre_integration->gyr_1;                    // t_j 处
            n_inliers += (int)pd.fis.size();
            pairs.push_back(std::move(pd));
        }
    }
    d.n_pairs = (int)pairs.size();
    d.n_inliers = n_inliers;

    // ---- 数据量检查：不足 → EXTEND（不涉及阈值更新）----
    if (d.n_pairs < params.min_pairs || d.n_inliers < params.min_inliers)
    {
        std::cout << "[后检测] 数据量不足: 帧对=" << d.n_pairs << "(需≥" << params.min_pairs
                  << ") 内点=" << d.n_inliers << "(需≥" << params.min_inliers
                  << ") → EXTEND_WINDOW" << std::endl;
        if (diag) *diag = d;
        return PostCheckVerdict::EXTEND_WINDOW;
    }

    // ---- 3. 残差评估（解处）----
    auto eval_residuals = [&](const PairList &pl, const Eigen::Vector3d &bg_lin,
                              const Eigen::Vector3d &bg_val,
                              const Eigen::Quaterniond &c_q, double c_td,
                              std::vector<double> &res) {
        res.clear();
        res.reserve(pl.size());
        for (const auto &pd : pl)
        {
            DocRotationCostFunctor cf(pd.fis, pd.fjs, pd.q_total, pd.J_bg, bg_lin,
                                      pd.w_start_raw, pd.w_end_raw, eps_lambda);
            double pbg[3] = {bg_val(0), bg_val(1), bg_val(2)};
            double pq[4]  = {c_q.x(), c_q.y(), c_q.z(), c_q.w()};
            double ptd[1] = {c_td};
            double r[1];
            cf(pbg, pq, ptd, r);
            res.push_back(r[0]);
        }
    };

    std::vector<double> r0;
    eval_residuals(pairs, Bg, Bg, q_bc, td, r0);
    std::vector<double> r_sorted = r0;
    std::sort(r_sorted.begin(), r_sorted.end());
    auto pct = [&](double p) { return r_sorted[(size_t)(p * (r_sorted.size() - 1))]; };
    double mean = 0.0;
    for (double r : r0) mean += r;
    mean /= (double)r0.size();
    double median = pct(0.5);
    double rms0 = 0.0;
    for (double r : r0) rms0 += r * r;
    rms0 = std::sqrt(rms0 / (double)r0.size());
    d.residual_mean = mean;
    d.residual_median = median;
    d.sigma_noise = median;   // 噪声地板 = 稳健中位数
    d.rms0 = rms0;
    std::cout << "[后检测] 帧对=" << d.n_pairs << " 内点=" << d.n_inliers
              << " | 解处残差: 均值=" << mean << " 中位数=" << median
              << " p90=" << pct(0.9) << " rms=" << rms0 << std::endl;

    // ---- 4. 平台边界：解处残差极小（完美拟合）→ 直接 ACCEPT（确认性）----
    if (rms0 <= 10.0 * std::sqrt(std::max(eps_lambda, 1e-30)))
    {
        std::cout << "[后检测] 解处残差进入 ελ 平台（完美拟合）→ ACCEPT（确认性）" << std::endl;
        d.opt_healthy = true;
        if (diag) *diag = d;
        return PostCheckVerdict::ACCEPT;
    }

    // ---- 5. 优化健康度：mean < health_ratio × median 才算健康 ----
    d.opt_healthy = (mean < pc.health_ratio * median);
    std::cout << "[后检测] 优化健康度: mean/median=" << (mean / std::max(median, 1e-30))
              << " (阈值 " << pc.health_ratio << ") → "
              << (d.opt_healthy ? "健康" : "不健康") << std::endl;
    if (!d.opt_healthy)
    {
        // 残差未降到噪声地板（局部极小/外点污染）→ σ̂ 虚高不可靠 → 不更新阈值
        std::cout << "[后检测] 结论: OPT_FAILED（优化未收敛/外点，σ̂ 不可靠，"
                  << "不更新阈值，延长窗口或换初值重试）" << std::endl;
        if (diag) *diag = d;
        return PostCheckVerdict::OPT_FAILED;
    }

    // ---- 6. 中心差分 Jacobian（解处，bg 只扰动 bg_val、bg_lin 固定）----
    std::vector<int> est_dims = {0, 1, 2};
    std::vector<int> bg_cols  = {0, 1, 2};
    std::vector<int> rbc_cols;
    std::vector<int> td_col;
    if (estimate_Rbc) { rbc_cols = {3, 4, 5}; est_dims.insert(est_dims.end(), rbc_cols.begin(), rbc_cols.end()); }
    if (estimate_td)  { td_col = {6}; est_dims.push_back(6); }
    const int n_param = (int)est_dims.size();
    const int n_res   = (int)pairs.size();
    d.n_param = n_param;

    const double h = 1e-6;
    Eigen::MatrixXd J(n_res, n_param);
    for (size_t k = 0; k < est_dims.size(); ++k)
    {
        const int dd = est_dims[k];
        std::vector<double> rp, rm;
        if (dd < 3)
        {
            Eigen::Vector3d bgp = Bg, bgm = Bg;
            bgp(dd) += h; bgm(dd) -= h;
            eval_residuals(pairs, Bg, bgp, q_bc, td, rp);
            eval_residuals(pairs, Bg, bgm, q_bc, td, rm);
        }
        else if (dd < 6)
        {
            Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
            if (dd == 4) axis = Eigen::Vector3d::UnitY();
            if (dd == 5) axis = Eigen::Vector3d::UnitZ();
            const Eigen::Quaterniond qp = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(h, axis));
            const Eigen::Quaterniond qm = q_bc * Eigen::Quaterniond(Eigen::AngleAxisd(-h, axis));
            eval_residuals(pairs, Bg, Bg, qp, td, rp);
            eval_residuals(pairs, Bg, Bg, qm, td, rm);
        }
        else
        {
            eval_residuals(pairs, Bg, Bg, q_bc, td + h, rp);
            eval_residuals(pairs, Bg, Bg, q_bc, td - h, rm);
        }
        for (int i = 0; i < n_res; ++i)
            J(i, (int)k) = (rp[i] - rm[i]) / (2.0 * h);
    }

    // ---- 7. §5 无量纲缩放 A_s = J·S，S = diag(ε_b I3, ε_R I3, ε_t) ----
    Eigen::MatrixXd A_s = J;
    for (size_t k = 0; k < est_dims.size(); ++k)
    {
        const int dd = est_dims[k];
        const double s = (dd < 3) ? params.eps_b : ((dd < 6) ? params.eps_R : params.eps_t);
        A_s.col((int)k) *= s;
    }
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_s, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd s = svd.singularValues();
        const double s_max = (s.size() > 0) ? s(0) : 0.0;
        for (int i = 0; i < (int)s.size(); ++i)
            if (s(i) > params.svd_rel_thresh * s_max) ++d.rank_A;
    }

    // ---- 8. §7/§8 有效局部曲率 + §8 敏感度半径 + §9 曲率独立率（同预检测）----
    // td: q = t_d，n = [bg, θbc]（§8.1）
    if (estimate_td)
    {
        std::vector<int> ncols = bg_cols;
        ncols.insert(ncols.end(), rbc_cols.begin(), rbc_cols.end());
        const Eigen::MatrixXd Htd = effectiveCurvature(A_s, td_col, ncols,
                                                       params.svd_rel_thresh, nullptr);
        d.kappa_td = Htd(0, 0);
        const double raw_td = A_s.col(td_col[0]).squaredNorm();
        d.eta_td = independenceRateScalar(d.kappa_td, raw_td);
        d.rho_td = 1.0 / std::sqrt(std::max(d.kappa_td, 1e-30));
        std::cout << "[后检测] td: κ=" << d.kappa_td
                  << " | ρ_td=" << d.rho_td << " | η_td=" << d.eta_td << std::endl;
    }
    // Rbc: q = θbc，n = [bg, t_d]（§8.2）
    if (estimate_Rbc)
    {
        std::vector<int> ncols = bg_cols;
        ncols.insert(ncols.end(), td_col.begin(), td_col.end());
        const Eigen::MatrixXd Hbc = effectiveCurvature(A_s, rbc_cols, ncols,
                                                       params.svd_rel_thresh, nullptr);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hbc);
        d.mu_bc = es.eigenvalues();                     // 升序
        for (int k = 0; k < 3; ++k)
            d.rho_bc(k) = 1.0 / std::sqrt(std::max(d.mu_bc(k), 1e-30));
        Eigen::MatrixXd Aqbc(n_res, 3);
        for (int c = 0; c < 3; ++c) Aqbc.col(c) = A_s.col(rbc_cols[c]);
        d.eta_bc_min = independenceRateMin(Aqbc, Hbc, params.svd_rel_thresh);
        std::cout << "[后检测] Rbc: ρ_bc=" << d.rho_bc.transpose()
                  << " | η_bc,min=" << d.eta_bc_min << std::endl;
    }
    // bg: q = b_g，n = [θbc, t_d]（§8.3）
    {
        std::vector<int> ncols = rbc_cols;
        ncols.insert(ncols.end(), td_col.begin(), td_col.end());
        const Eigen::MatrixXd Hbg = effectiveCurvature(A_s, bg_cols, ncols,
                                                       params.svd_rel_thresh, nullptr);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hbg);
        d.nu_bg = es.eigenvalues();                     // 升序
        for (int k = 0; k < 3; ++k)
            d.rho_bg(k) = 1.0 / std::sqrt(std::max(d.nu_bg(k), 1e-30));
        Eigen::MatrixXd Aqbg(n_res, 3);
        for (int c = 0; c < 3; ++c) Aqbg.col(c) = A_s.col(bg_cols[c]);
        d.eta_bg_min = independenceRateMin(Aqbg, Hbg, params.svd_rel_thresh);
        std::cout << "[后检测] bg: ρ_bg=" << d.rho_bg.transpose()
                  << " | η_bg,min=" << d.eta_bg_min << std::endl;
    }

    // ---- 9. 自适应阈值 ρ_max = clamp(c/σ̂_noise, rho_min, rho_max_bound) ----
    d.rho_max_adaptive = std::min(pc.rho_max_bound,
                                  std::max(pc.rho_min,
                                           pc.safety_factor / std::max(d.sigma_noise, 1e-30)));

    // ---- 10. 解处判定：ρ ≤ ρ_max_adaptive 且 η ≥ η0 ----
    bool pass = true;
    if (estimate_td  && d.rho_td > d.rho_max_adaptive) pass = false;
    if (estimate_Rbc && d.rho_bc.maxCoeff() > d.rho_max_adaptive) pass = false;
    if (params.gate_bg && d.rho_bg.maxCoeff() > d.rho_max_adaptive) pass = false;
    if (estimate_td  && d.eta_td < params.eta0) pass = false;
    if (estimate_Rbc && d.eta_bc_min < params.eta0) pass = false;
    if (params.gate_bg && d.eta_bg_min < params.eta0) pass = false;

    std::cout << "[后检测] 自适应阈值 ρ_max=" << d.rho_max_adaptive
              << " (=c/σ̂_noise=" << pc.safety_factor << "/" << d.sigma_noise
              << ", c=" << pc.safety_factor << ")"
              << " | ρ_td=" << d.rho_td
              << " | maxρ_bc=" << d.rho_bc.maxCoeff()
              << " | maxρ_bg=" << d.rho_bg.maxCoeff()
              << " | η_td=" << d.eta_td << " η_bc,min=" << d.eta_bc_min
              << " η_bg,min=" << d.eta_bg_min << std::endl;
    std::cout << "[后检测] 结论: " << (pass ? "ACCEPT → 接受旋转初始化"
                                            : "EXTEND_WINDOW → 更新 ρ_max 并延长窗口")
              << std::endl;

    if (diag) *diag = d;
    return pass ? PostCheckVerdict::ACCEPT : PostCheckVerdict::EXTEND_WINDOW;
}
