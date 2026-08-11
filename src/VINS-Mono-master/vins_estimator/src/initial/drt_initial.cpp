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

// ============================================================================
// 已知旋转下基于极线约束的帧间平移计算（相机平移 Tc0c 估计）
// 文档：fixed_rotation_camera_translation_initialization_corrected.md §4-§7
// ----------------------------------------------------------------------------
// 输入：all_image_frame —— 每帧的相机旋转 frame.Rc0c（^c0 R_ci）已由旋转优化得到。
// 输出：写回 frame.Tc0c（= ^c0 p_ci，视觉尺度下的相机中心位置）、frame.R、frame.T。
// 返回：true = 成功写出位置；false = 数据不足 / 退化。
// 步骤（对应文档）：
//   1) 建立有序帧表 R[i]（第0帧为参考，必要时左乘 R0^T 归一化）      §3.2
//   2) 多跨度帧对收集共视 bearing，M_ij 谱诊断过滤                  §4.2/§6
//   3) 全局位置系统 A_p x_p = 0，H_p = A_p^T A_p 最小特征向量        §5.1
//   4) 用基线最大的帧对归一化视觉尺度                                §5.2
//   5) 二视图正深度检查消解整体符号二义性                            §5.3
//   6) 写回 Rc0c / Tc0c / R / T                                     §7
// 注意：单目极线约束只恢复"视觉尺度"下的位置；米制尺度由后续
//       VisualIMUAlignment 恢复，本函数不触碰 TIC[0]（§8）。
//       阈值（tau_ratio / tau_energy）应按数据标定（§6），默认值较宽松。
// ============================================================================
bool translationEstimator(std::map<double, ImageFrame> &all_image_frame,
                          int min_features,
                          int min_pair_step,
                          int max_pair_step,
                          double tau_ratio,
                          double tau_energy,
                          bool verbose)
{
    if (min_pair_step < 1) min_pair_step = 1;
    if (min_pair_step > max_pair_step) min_pair_step = max_pair_step;

    TicToc tt;
    std::cout << "\n========== 已知旋转下的帧间平移计算 (§4-§7) ==========" << std::endl;

    if (all_image_frame.size() < 2)
    {
        std::cout << "[平移] 帧数不足（" << all_image_frame.size()
                  << "），无法计算" << std::endl;
        return false;
    }

    // ---- 1. 按时间序建立有序帧表，R[i] = ^c0 R_ci ----
    std::vector<const ImageFrame *> frames;
    std::vector<Eigen::Matrix3d> R;
    frames.reserve(all_image_frame.size());
    R.reserve(all_image_frame.size());
    for (const auto &kv : all_image_frame)
    {
        frames.push_back(&kv.second);
        R.push_back(kv.second.Rc0c);
    }
    const int N = (int)frames.size();

    // 第0帧为视觉参考系：R[0] 应为单位阵；若旋转优化未归一化则左乘 R0^T（§3.2）
    const double err0 = (R[0] - Eigen::Matrix3d::Identity()).norm();
    if (err0 > 1e-6)
    {
        if (verbose)
            std::cout << "[平移] 第0帧 Rc0c 非单位阵（偏差 " << err0
                      << "），左乘 R0^T 归一化到参考系" << std::endl;
        const Eigen::Matrix3d R0t = R[0].transpose();
        for (int i = 0; i < N; ++i) R[i] = R0t * R[i];
    }

    // ---- 2. 多跨度帧对收集 + M_ij 谱诊断过滤（§4.2/§6）----
    // 注意：feature_tracker 发布的坐标为归一化平面坐标 [x/z, y/z, 1]（p.z=1），
    //       非单位向量；按文档 §9 伪代码显式 normalize 为单位 bearing（§2.2 的 f_ik），
    //       保证 M_ij 中每个特征等权（w=1），否则大模长特征会主导谱诊断。
    struct PairInfo {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int i, j;                               // 帧索引
        std::vector<Eigen::Vector3d> fis, fjs;  // 共视单位 bearing（帧 i / 帧 j）
    };
    std::vector<PairInfo> pairs;

    for (int i = 0; i < N; ++i)
    {
        for (int step = min_pair_step; step <= max_pair_step; ++step)
        {
            const int j = i + step;
            if (j >= N) break;

            const ImageFrame &frame_i = *frames[i];
            const ImageFrame &frame_j = *frames[j];

            std::vector<Eigen::Vector3d> fis, fjs;
            for (const auto &fp : frame_i.points)
            {
                auto fjn = frame_j.points.find(fp.first);
                if (fjn != frame_j.points.end())
                {
                    fis.push_back(fp.second[0].second.head<3>().normalized());
                    fjs.push_back(fjn->second[0].second.head<3>().normalized());
                }
            }
            const int nf = (int)fis.size();
            if (nf < min_features) continue;

            // M_ij = Σ_k n_ij^k (n_ij^k)^T（§4.3，权重 w=1）
            // n_ij^k = [f_ik]_× R_ij f_jk，R_ij = R_i^T R_j
            const Eigen::Matrix3d Rij = R[i].transpose() * R[j];
            Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
            for (int k = 0; k < nf; ++k)
            {
                const Eigen::Vector3d n = fis[k].cross(Rij * fjs[k]);
                M += n * n.transpose();
            }
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(M);
            const Eigen::Vector3d lam = es.eigenvalues();   // λ1≤λ2≤λ3 升序
            const double traceM = M.trace();
            const double rho = lam(0) / (lam(1) + 1e-30);   // §6 eq.17 谱比
            if (traceM < 1e-12) continue;                   // 纯旋转 / 视差过小
            if (lam(1) < tau_energy) continue;              // q_ij = λ2 能量不足
            if (rho > tau_ratio) continue;                  // 谱比不满足

            PairInfo pi;
            pi.i = i; pi.j = j;
            pi.fis = std::move(fis);
            pi.fjs = std::move(fjs);
            pairs.push_back(std::move(pi));
        }
    }

    if (pairs.empty())
    {
        std::cout << "[平移] 无通过谱诊断的帧对（min_features=" << min_features
                  << " tau_ratio=" << tau_ratio
                  << " tau_energy=" << tau_energy << "），无法计算" << std::endl;
        return false;
    }

    // ---- 3. 全局位置系统（§5.1）：(a_ij^k)^T (p_j - p_i) = 0，p_0 = 0 ----
    // a_ij^k = R_i n_ij^k = (R_i f_ik) × (R_j f_jk)
    // 未知量 x_p = [p_1^T ... p_{N-1}^T]^T，共 3(N-1) 维；直接累加 H_p = A_p^T A_p
    const int dim = 3 * (N - 1);
    Eigen::MatrixXd Hp = Eigen::MatrixXd::Zero(dim, dim);
    int n_rows = 0;
    for (const auto &pi : pairs)
    {
        const int bi = (pi.i > 0) ? 3 * (pi.i - 1) : -1;   // p_i 列块起点（p_0 无列）
        const int bj = (pi.j > 0) ? 3 * (pi.j - 1) : -1;   // p_j 列块起点
        for (int k = 0; k < (int)pi.fis.size(); ++k)
        {
            const Eigen::Vector3d a = (R[pi.i] * pi.fis[k]).cross(R[pi.j] * pi.fjs[k]);
            if (bi >= 0) Hp.block<3, 3>(bi, bi) += a * a.transpose();
            if (bj >= 0) Hp.block<3, 3>(bj, bj) += a * a.transpose();
            if (bi >= 0 && bj >= 0)
            {
                Hp.block<3, 3>(bi, bj) -= a * a.transpose();
                Hp.block<3, 3>(bj, bi) -= a * a.transpose();
            }
            ++n_rows;
        }
    }
    if (n_rows < dim)
    {
        std::cout << "[平移] 约束行数 " << n_rows << " < 未知量 " << dim
                  << "，系统欠定" << std::endl;
        return false;
    }

    // 最小特征向量（= A_p 最小奇异值的右奇异向量，§5.1 eq.13）
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> esH(Hp);
    const double lam0 = esH.eigenvalues()(0);   // 最小（尺度零空间应≈0）
    const double lam1 = esH.eigenvalues()(1);   // 次小
    const double lam_max = esH.eigenvalues()(dim - 1);
    // 退化检查（§6 谱隙精神）：若 λ1 相对过小，说明位置系统存在 >1 维零空间
    // （纯旋转 / 特征配置退化 / 帧对图不连通），平移方向不可观，直接拒绝。
    if (lam1 < 1e-10 * std::max(lam_max, 1e-30))
    {
        std::cout << "[平移] 位置系统退化：λ1=" << lam1
                  << " 相对 λmax=" << lam_max << " 过小，平移不可观" << std::endl;
        return false;
    }
    // 警告：尺度零空间不干净（λ0 相对 λ1 过大）→ 最小特征向量被噪声污染
    if (lam0 > 0.1 * std::max(lam1, 1e-30))
        std::cout << "[平移] 警告：尺度零空间不干净（λ0/λ1="
                  << lam0 / std::max(lam1, 1e-30)
                  << "），解受噪声影响，结果仅供参考" << std::endl;
    Eigen::VectorXd x = esH.eigenvectors().col(0);   // 特征值升序 → 第0列

    std::vector<Eigen::Vector3d> p(N, Eigen::Vector3d::Zero());
    for (int i = 1; i < N; ++i)
        p[i] = Eigen::Vector3d(x.segment<3>(3 * (i - 1)));

    // ---- 4. 尺度归一化：选择基线最大的帧对（§5.2 eq.14）----
    int rs = 0, ss = 0;
    double best_base = -1.0;
    for (const auto &pi : pairs)
    {
        const double base = (p[pi.j] - p[pi.i]).norm();
        if (base > best_base) { best_base = base; rs = pi.i; ss = pi.j; }
    }
    if (best_base < 1e-12)
    {
        std::cout << "[平移] 视觉基线过小，运动退化" << std::endl;
        return false;
    }
    const double beta = 1.0 / best_base;
    for (int i = 0; i < N; ++i) p[i] *= beta;

    // ---- 5. 二视图正深度检查消解整体符号（§5.3 eq.16）----
    // 对符号 s：t_ij = R_i^T (s·(p_j - p_i))，[f_ik, -R_ij f_jk][λ_ik; λ_jk] = t_ij
    auto countPositiveDepth = [&](double sign) {
        int pos = 0, total = 0;
        for (const auto &pi : pairs)
        {
            const int i = pi.i, j = pi.j;
            const Eigen::Matrix3d Rij = R[i].transpose() * R[j];
            const Eigen::Vector3d tij = R[i].transpose() * (sign * (p[j] - p[i]));
            for (int k = 0; k < (int)pi.fis.size(); ++k)
            {
                // 跳过近零视差匹配：f_ik ∥ R_ij f_jk 时 2×2 系统奇异，
                // 深度不可靠且 LDLT 会输出垃圾值污染符号统计（§5.3"选定匹配"）。
                const double cosang = pi.fis[k].dot(Rij * pi.fjs[k]);
                if (cosang > 0.999) continue;   // 夹角 < ~2.6°
                Eigen::Matrix<double, 3, 2> A;
                A.col(0) = pi.fis[k];
                A.col(1) = -Rij * pi.fjs[k];
                const Eigen::Vector2d lam =
                    (A.transpose() * A).ldlt().solve(A.transpose() * tij);
                if (lam(0) > 0.0 && lam(1) > 0.0) ++pos;
                ++total;
            }
        }
        return std::make_pair(pos, total);
    };
    const auto cp = countPositiveDepth(1.0);
    const auto cm = countPositiveDepth(-1.0);
    const double sgn = (cm.first > cp.first) ? -1.0 : 1.0;
    if (sgn < 0.0)
        for (int i = 0; i < N; ++i) p[i] *= -1.0;

    // ---- 6. 写回（§7）----
    int idx = 0;
    for (auto &kv : all_image_frame)
    {
        ImageFrame &f = kv.second;
        f.Rc0c = R[idx];                        // ^c0 R_ci
        f.Tc0c = p[idx];                        // ^c0 p_ci（视觉尺度）
        f.R    = R[idx] * RIC[0].transpose();   // ^c0 R_bi = ^c0 R_ci * R_cb
        f.T    = p[idx];                        // 相机中心，视觉尺度
        ++idx;
    }

    if (verbose)
    {
        std::cout << "[平移] 帧数=" << N
                  << " 帧对=" << pairs.size()
                  << " 约束行=" << n_rows
                  << " | H_p 特征值 λ0=" << lam0 << " λ1=" << lam1
                  << " | 归一化基线帧对 (" << rs << "," << ss << ")"
                  << " | 符号=" << (sgn > 0 ? "+" : "-")
                  << "（正深度 +:" << cp.first << "/" << cp.second
                  << "  -:" << cm.first << "/" << cm.second << "）"
                  << " | 耗时=" << tt.toc() << "ms" << std::endl;
        std::cout << "[平移] 相机位置 (视觉尺度):";
        for (int i = 0; i < N; ++i)
            std::cout << "  p" << i << "=" << p[i].transpose();
        std::cout << std::endl;
    }

    return true;
}

// ============================================================================
// 消去速度后的尺度、重力与加速度计零偏闭式解算
// 文档：closed_form_scale_gravity_accel_bias_velocity_eliminated_规范化审核版.md
// ----------------------------------------------------------------------------
// 状态 x = [s, (g^c0)^T, (δb_a)^T]^T ∈ R^7（文档 eq.1）
// 输入约定（文档 §2，已在本工程核对）：
//   frame.R  = ^c0 R_bi（body 旋转）              = R_{b_k}^{c0}
//   frame.T  = 视觉尺度下的相机中心                 = P_{c_k}^{c0}
//   TIC[0]   = t_bc（body 原点→camera 原点，body 系）
//   pre_integration->delta_p / delta_v             = ΔP^0 / ΔV^0（最终 bg 重积分后）
//   pre_integration->jacobian(O_P/O_V, O_BA)       = J_{b_a}^P / J_{b_a}^V
//   pre_integration->sum_dt                        = Δt
// 重力符号：本函数按文档输出"物理重力" g^c0（向下）；源码补偿向量
//   G = [0,0,9.8] = g_code，满足 g^c0 = -g_code（接口处仅一次映射，§1 注）。
// 零偏：绝对零偏 ba = bar_ba + δb_a（eq.43），bar_ba 取预积分线性化点。
// 方法（文档 §4-§7）：k=0..N-3 由位置方程消去速度 v_k^{b_k} = E_k x + e_k
//   （eq.20/21），代入速度方程得 A_k x = b_k（eq.44），堆叠 3(N-2)×7 系统；
//   列主元 QR + SVD 最小二乘求解（不显式构造正规方程，§7），
//   并做秩/条件数/尺度正性验收检查（§7.1/§10）。
// 返回：true = 解算成功；false = 数据不足 / 不可观 / 条件数超限 / 尺度非正。
// ============================================================================
bool scaleGravityBiasEstimator(std::map<double, ImageFrame> &all_image_frame,
                               double &s,
                               Eigen::Vector3d &g_c0,
                               Eigen::Vector3d &ba,
                               double tau_gravity,
                               double tau_kappa,
                               bool verbose)
{
    TicToc tt;
    std::cout << "\n========== 闭式解算尺度/重力/加计零偏（消去速度，文档 §4-§7） ==========" << std::endl;

    const int N = (int)all_image_frame.size();
    // 数据量必要条件：3(N-2) ≥ 7 → N ≥ 5（文档 §7.1 eq.38）
    if (N < 5)
    {
        std::cout << "[尺度重力] 帧数不足（" << N << " < 5），无法解算" << std::endl;
        return false;
    }

    // 有序帧表
    std::vector<const ImageFrame *> frames;
    frames.reserve(N);
    for (const auto &kv : all_image_frame)
        frames.push_back(&kv.second);

    for (int k = 0; k < N; ++k)
        if (!frames[k]->pre_integration)
        {
            std::cout << "[尺度重力] 帧 " << k << " 无预积分，无法解算" << std::endl;
            return false;
        }

    // ---- 构造 A_red (3(N-2) × 7) / b_red（文档 eq.36/44）----
    const int n_rows = 3 * (N - 2);
    Eigen::MatrixXd A(n_rows, 7);
    Eigen::VectorXd b(n_rows);
    const Eigen::Vector3d tbc = TIC[0];

    int row = 0;
    for (int k = 0; k <= N - 3; ++k)
    {
        const ImageFrame &fk  = *frames[k];
        const ImageFrame &fk1 = *frames[k + 1];
        const ImageFrame &fk2 = *frames[k + 2];

        IntegrationBase *pk  = fk .pre_integration;   // [t_k,     t_{k+1}]
        IntegrationBase *pk1 = fk1.pre_integration;   // [t_{k+1}, t_{k+2}]

        const double dtk  = pk ->sum_dt;
        const double dtk1 = pk1->sum_dt;
        if (dtk <= 0.0 || dtk1 <= 0.0)
        {
            std::cout << "[尺度重力] 帧 " << k << " 预积分时间间隔非正，无法解算" << std::endl;
            return false;
        }

        // 旋转与相机中心（eq.2/eq.5）
        const Eigen::Matrix3d Rk   = fk .R;                    // R_{b_k}^{c0}
        const Eigen::Matrix3d Rk1  = fk1.R;                    // R_{b_{k+1}}^{c0}
        const Eigen::Matrix3d Rk2  = fk2.R;                    // R_{b_{k+2}}^{c0}
        const Eigen::Matrix3d R_k1_k  = Rk .transpose() * Rk1; // R_{b_{k+1}}^{b_k}
        const Eigen::Matrix3d R_k2_k1 = Rk1.transpose() * Rk2; // R_{b_{k+2}}^{b_{k+1}}
        const Eigen::Vector3d Pck  = fk .T;                    // P_{c_k}^{c0}
        const Eigen::Vector3d Pck1 = fk1.T;                    // P_{c_{k+1}}^{c0}
        const Eigen::Vector3d Pck2 = fk2.T;                    // P_{c_{k+2}}^{c0}

        // 名义预积分量与零偏雅可比（eq.6）
        const Eigen::Vector3d dPk  = pk ->delta_p;
        const Eigen::Vector3d dPk1 = pk1->delta_p;
        const Eigen::Vector3d dVk  = pk ->delta_v;
        const Eigen::Vector3d dVk1 = pk1->delta_v;
        const Eigen::Matrix3d JbaPk  = pk ->jacobian.block<3, 3>(O_P, O_BA);
        const Eigen::Matrix3d JbaPk1 = pk1->jacobian.block<3, 3>(O_P, O_BA);
        const Eigen::Matrix3d JbaVk  = pk ->jacobian.block<3, 3>(O_V, O_BA);
        const Eigen::Matrix3d JbaVk1 = pk1->jacobian.block<3, 3>(O_V, O_BA);

        // ---- 第 k 区间（k→k+1）：E_k / e_k（eq.14/17/18/20/21/23/25）----
        const Eigen::Vector3d dk   = Rk.transpose() * (Pck1 - Pck);      // eq.17
        const Eigen::Vector3d rPk  = dPk + R_k1_k * tbc - tbc;           // eq.14
        const Eigen::Vector3d rVk  = dVk;                                // eq.23
        const Eigen::Matrix3d Qk   = -0.5 * dtk * dtk * Rk.transpose();  // eq.18
        Eigen::Matrix<double, 3, 7> Ek;
        Ek.setZero();                                                    // eq.20/21
        Ek.block<3, 1>(0, 0) = dk;          // 尺度列
        Ek.block<3, 3>(0, 1) = Qk;          // 重力列
        Ek.block<3, 3>(0, 4) = -JbaPk;      // 零偏列
        Ek /= dtk;
        const Eigen::Vector3d ek = -rPk / dtk;                           // eq.21
        Eigen::Matrix<double, 3, 7> HkV;
        HkV.setZero();                                                   // eq.25
        HkV.block<3, 3>(0, 1) = -dtk * Rk.transpose();
        HkV.block<3, 3>(0, 4) = -JbaVk;

        // ---- 第 k+1 区间（k+1→k+2）：E_{k+1} / e_{k+1} ----
        const Eigen::Vector3d dk1  = Rk1.transpose() * (Pck2 - Pck1);
        const Eigen::Vector3d rPk1 = dPk1 + R_k2_k1 * tbc - tbc;
        const Eigen::Matrix3d Qk1  = -0.5 * dtk1 * dtk1 * Rk1.transpose();
        Eigen::Matrix<double, 3, 7> Ek1;
        Ek1.setZero();
        Ek1.block<3, 1>(0, 0) = dk1;
        Ek1.block<3, 3>(0, 1) = Qk1;
        Ek1.block<3, 3>(0, 4) = -JbaPk1;
        Ek1 /= dtk1;
        const Eigen::Vector3d ek1 = -rPk1 / dtk1;

        // ---- 核心方程（eq.27/28 = eq.44）----
        const Eigen::Matrix<double, 3, 7> Ak = -Ek + R_k1_k * Ek1 + HkV;
        const Eigen::Vector3d bk = rVk + ek - R_k1_k * ek1;

        A.block<3, 7>(row, 0) = Ak;
        b.segment<3>(row) = bk;
        row += 3;
    }

    // ---- 最小二乘解：列主元 QR（求解）+ SVD（条件数），不显式构造正规方程（§7）----
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
    if (qr.rank() < 7)   // 文档 §7.1 eq.39：rank 必须为 7
    {
        std::cout << "[尺度重力] rank(A_red)=" << qr.rank()
                  << " < 7，尺度/重力/零偏不可观（运动激励不足）" << std::endl;
        return false;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd sv = svd.singularValues();
    const double cond = sv(0) / std::max(sv(6), 1e-30);
    if (cond > tau_kappa)
    {
        std::cout << "[尺度重力] 条件数 " << cond << " > " << tau_kappa
                  << "，解不可靠" << std::endl;
        return false;
    }
    const Eigen::VectorXd x = qr.solve(b);

    // ---- 提取状态（§10）----
    s    = x(0);
    g_c0 = x.segment<3>(1);
    const Eigen::Vector3d dba = x.segment<3>(4);
    // 绝对零偏：ba = bar_ba + δb_a（eq.43），bar_ba 为预积分线性化点
    const Eigen::Vector3d bar_ba = frames[0]->pre_integration->linearized_ba;
    ba = bar_ba + dba;

    // ---- 验收检查（§10）----
    bool ok = true;
    if (s <= 0.0)
    {
        std::cout << "[尺度重力] 尺度 s=" << s << " ≤ 0，结果无效" << std::endl;
        ok = false;
    }
    const double g_norm = g_c0.norm();
    if (std::fabs(g_norm - G.norm()) > tau_gravity)
        std::cout << "[尺度重力] 警告: ||g^c0||=" << g_norm
                  << " 偏离 G=" << G.norm() << " 超过 " << tau_gravity
                  << " m/s²（结果仅供参考）" << std::endl;

    if (verbose)
    {
        const double rms = (A * x - b).norm() / std::sqrt((double)n_rows);
        std::cout << "[尺度重力] N=" << N << " 行=" << n_rows
                  << " | rank=" << qr.rank() << " 条件数=" << cond
                  << " | 残差 rms=" << rms
                  << " | 耗时=" << tt.toc() << "ms" << std::endl;
        std::cout << "[尺度重力] s=" << s
                  << " | g^c0=" << g_c0.transpose() << " (|g|=" << g_norm << ")"
                  << " | ba=" << ba.transpose()
                  << " | (g_code = -g^c0 = " << (-g_c0).transpose() << ")" << std::endl;
    }

    return ok;
}

// ============================================================================
// 视觉-惯性对齐：旋转优化结果 → 矫正预积分 → Rc0c → Tc0c → 尺度/重力/零偏
// 文档：fixed_rotation_camera_translation_initialization_corrected.md §3/§9
//       + closed_form_scale_gravity_accel_bias_velocity_eliminated_规范化审核版.md
// ----------------------------------------------------------------------------
// 输入：all_image_frame（含预积分与共视特征）；旋转优化估计的 bg / q_bc(Rbc) / td。
// 步骤（对应文档）：
//   1) 用 bg 矫正 IMU 预积分（repropagate，§6.2）
//   2) 用优化后的 Rbc 逐帧累积相机旋转 Rc0c（§3.2/§3.3）
//   3) 调用 translationEstimator 计算所有帧的相机平移 Tc0c（阈值更严格）
//   4) 调用 scaleGravityBiasEstimator 闭式解算尺度/物理重力/加计零偏
// 注意：当前工程预积分不含 td（td 仅用于旋转 cost functor 的像素域补偿），
//       按文档 §3.1/§3.3 此处应置 td=0，直接用 repropagate 后的 delta_q。
// 可选输出：s_out / g_c0_out / ba_out（scaleGravityBiasEstimator 的结果）。
// ============================================================================
void VisionIMuAlignment(std::map<double, ImageFrame> &all_image_frame,
                        Eigen::Vector3d bg,
                        Eigen::Quaterniond q_bc,
                        double td,
                        double *s_out,
                        Eigen::Vector3d *g_c0_out,
                        Eigen::Vector3d *ba_out)
{
    std::cout << "\n========== Vision-IMU 对齐：矫正预积分 → Rc0c → Tc0c ==========" << std::endl;

    // 确保全局外参与旋转优化估计一致（translationEstimator 写回 frame.R 时用 RIC[0]）
    RIC[0] = q_bc.toRotationMatrix();

    if (std::fabs(td) > 1e-9)
        std::cout << "[VisionIMuAlignment] 提示: td=" << td
                  << " 未纳入预积分模型（当前工程仅像素域补偿），"
                  << "Rc0c 按 td=0 计算（文档 §3.1/§3.3）" << std::endl;

    // ---- 1. 用旋转优化估计的 bg 矫正 IMU 预积分（repropagate，§6.2）----
    for (auto &frame_it : all_image_frame)
        if (frame_it.second.pre_integration)
            frame_it.second.pre_integration->repropagate(Eigen::Vector3d::Zero(), bg);

    // ---- 2. 逐帧累积相机旋转 Rc0c（§3.2/§3.3）----
    // Rc0c_0 = I；Rc0c_j = Rc0c_i * R_cb * delta_q_ij * R_bc
    // delta_q_ij = ^b_i R_b_j（repropagate 后的预积分旋转）
    const Eigen::Matrix3d R_bc = q_bc.toRotationMatrix();   // camera -> body（优化估计）
    const Eigen::Matrix3d R_cb = R_bc.transpose();          // body -> camera

    auto frame_prev = all_image_frame.begin();
    frame_prev->second.Rc0c = Eigen::Matrix3d::Identity();  // 第一帧为参考（§3.2 R0=I）
    for (auto frame_cur = std::next(frame_prev); frame_cur != all_image_frame.end(); ++frame_cur)
    {
        if (!frame_cur->second.pre_integration)
        {
            std::cout << "[VisionIMuAlignment] 警告: 帧 " << frame_cur->second.t
                      << " 无预积分，Rc0c 保持上一帧值" << std::endl;
            frame_cur->second.Rc0c = frame_prev->second.Rc0c;
            continue;
        }
        const Eigen::Matrix3d R_bi_bj =
            frame_cur->second.pre_integration->delta_q.toRotationMatrix();
        frame_cur->second.Rc0c = frame_prev->second.Rc0c * R_cb * R_bi_bj * R_bc;
        frame_prev = frame_cur;
    }

    // ---- 3. 极线约束计算相机平移 Tc0c（阈值更严格）----
    // 更严格（相对默认）：min_features 8→12（共视数更多），
    //   tau_ratio 0.9→0.5（谱比 λ1/λ2 更苛刻），
    //   tau_energy 1e-8→1e-6（λ2 能量门槛更高）。
    const bool ok = translationEstimator(all_image_frame,
                                         50,    // min_features（更严格）
                                         4,     // min_pair_step
                                         10,    // max_pair_step
                                         0.5,   // tau_ratio（更严格）
                                         1e-6,  // tau_energy（更严格）
                                         true); // verbose
    std::cout << "[VisionIMuAlignment] 平移估计 " << (ok ? "成功" : "失败/退化")
              << std::endl;

    // ---- 4. 闭式解算尺度/物理重力/加计零偏（消去速度，closed_form_..._规范化审核版.md）----
    double s = 0.0;
    Eigen::Vector3d g_c0 = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();
    const bool ok_sgb = scaleGravityBiasEstimator(all_image_frame, s, g_c0, ba,
                                                  0.5,   // tau_gravity（m/s²，仅告警）
                                                  1e8,   // tau_kappa（条件数上限）
                                                  true); // verbose
    std::cout << "[VisionIMuAlignment] 尺度/重力/零偏解算 "
              << (ok_sgb ? "成功" : "失败/退化") << std::endl;
    if (s_out)    *s_out    = s;
    if (g_c0_out) *g_c0_out = g_c0;
    if (ba_out)   *ba_out   = ba;
}


