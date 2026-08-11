#pragma once

#include <cmath>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Eigen>
#include <memory>

#include "../parameters.h"
#include "../utility/geometry.hpp"
#include "../utility/opengvMethod.hpp"
#include "../factor/integration_base.h"

struct BiasSolverCostFunctor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasSolverCostFunctor(const std::vector<Eigen::Vector3d> &bearings1,
                          const std::vector<Eigen::Vector3d> &bearings2,
                          const std::vector<Eigen::Vector2d> &velocities1,
                          const std::vector<Eigen::Vector2d> &velocities2,
                          const IntegrationBase &integrate) : 
            _bearings1(bearings1), _bearings2(bearings2),
            _velocities1(velocities1), _velocities2(velocities2){
        
        /*******************************/
        // jacobina_q_bg = integrate.JRg_;
        // qjk_imu = integrate.dR_.unit_quaternion();
        jacobina_q_bg = integrate.jacobian.block<3, 3>(O_R, O_BG);
        qjk_imu = integrate.delta_q;
        /*******************************/
    }

    BiasSolverCostFunctor(const std::vector<Eigen::Vector3d> &bearings1,
                          const std::vector<Eigen::Vector3d> &bearings2,
                          const std::vector<Eigen::Vector2d> &velocities1,
                          const std::vector<Eigen::Vector2d> &velocities2,
                          const Eigen::Matrix3d &accumulated_jacobian_bg,
                          const Eigen::Quaterniond &accumulated_delta_q) : 
            _bearings1(bearings1), _bearings2(bearings2),
            _velocities1(velocities1), _velocities2(velocities2),
            jacobina_q_bg(accumulated_jacobian_bg),
            qjk_imu(accumulated_delta_q) {}

    template<typename T>
    bool operator()(const T *const bg_ptr, const T *const q_bc_ptr, const T *const td_ptr, T *residual) const {
        T td = td_ptr[0];

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> deltaBg(bg_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_bc(q_bc_ptr);

        Eigen::Matrix<T, 3, 1> jacobian_bg = jacobina_q_bg.cast<T>() * deltaBg;
        Eigen::Matrix<T, 4, 1> qij_tmp;
        ceres::AngleAxisToQuaternion(jacobian_bg.data(), qij_tmp.data());
        Eigen::Quaternion<T> qij(qij_tmp(0), qij_tmp(1), qij_tmp(2), qij_tmp(3));

        Eigen::Matrix<T, 3, 3> xxF = Eigen::Matrix<T, 3, 3>::Zero();
        Eigen::Matrix<T, 3, 3> yyF = Eigen::Matrix<T, 3, 3>::Zero();
        Eigen::Matrix<T, 3, 3> zzF = Eigen::Matrix<T, 3, 3>::Zero();
        Eigen::Matrix<T, 3, 3> xyF = Eigen::Matrix<T, 3, 3>::Zero();
        Eigen::Matrix<T, 3, 3> yzF = Eigen::Matrix<T, 3, 3>::Zero();
        Eigen::Matrix<T, 3, 3> xzF = Eigen::Matrix<T, 3, 3>::Zero();

        for (int i = 0; i < _bearings1.size(); i++) {
            T u_1 = T(_bearings1[i].x() / _bearings1[i].z());
            T v_1 = T(_bearings1[i].y() / _bearings1[i].z());
            
            u_1 = u_1 + T(_velocities1[i].x()) * td;
            v_1 = v_1 + T(_velocities1[i].y()) * td;
            
            Eigen::Matrix<T, 3, 1> f1(u_1, v_1, T(1.0));
            f1.normalize();
            f1 = qjk_imu.inverse().cast<T>() * q_bc * f1; 

            T u_2 = T(_bearings2[i].x() / _bearings2[i].z());
            T v_2 = T(_bearings2[i].y() / _bearings2[i].z());
            
            u_2 = u_2 + T(_velocities2[i].x()) * td;
            v_2 = v_2 + T(_velocities2[i].y()) * td;

            Eigen::Matrix<T, 3, 1> f2(u_2, v_2, T(1.0));
            f2.normalize();
            f2 = q_bc * f2;

            Eigen::Matrix<T, 3, 3> F = f2 * f2.transpose();

            xxF = xxF + f1[0] * f1[0] * F;
            yyF = yyF + f1[1] * f1[1] * F;
            zzF = zzF + f1[2] * f1[2] * F;
            xyF = xyF + f1[0] * f1[1] * F;
            yzF = yzF + f1[1] * f1[2] * F;
            xzF = xzF + f1[0] * f1[2] * F;
        }

        Eigen::Matrix<T, 3, 1> cayley = Quaternion2Cayley<T>(qij); 
        Eigen::Matrix<T, 1, 3> jacobian;

        T EV = opengv::GetSmallestEVwithJacobian(xxF, yyF, zzF, xyF, yzF, xzF, cayley, jacobian);
        residual[0] = EV;

        return true;
    }

    static ceres::CostFunction *
    Create(const std::vector<Eigen::Vector3d> &bearings1,
           const std::vector<Eigen::Vector3d> &bearings2,
           const std::vector<Eigen::Vector2d> &velocities1,
           const std::vector<Eigen::Vector2d> &velocities2,
           const IntegrationBase &integratePtr) {
        
        return (new ceres::AutoDiffCostFunction<BiasSolverCostFunctor, 1, 3, 4, 1>(
                new BiasSolverCostFunctor(bearings1, bearings2, velocities1, velocities2, integratePtr)));
    }

    static ceres::CostFunction *
    CreateAccumulated(const std::vector<Eigen::Vector3d> &bearings1,
                      const std::vector<Eigen::Vector3d> &bearings2,
                      const std::vector<Eigen::Vector2d> &velocities1,
                      const std::vector<Eigen::Vector2d> &velocities2,
                      const Eigen::Matrix3d &accumulated_jacobian_bg,
                      const Eigen::Quaterniond &accumulated_delta_q) {
        
        return (new ceres::AutoDiffCostFunction<BiasSolverCostFunctor, 1, 3, 4, 1>(
                new BiasSolverCostFunctor(bearings1, bearings2, velocities1, velocities2,
                                          accumulated_jacobian_bg, accumulated_delta_q)));
    }

private:
   std::vector<Eigen::Vector3d> _bearings1;
   std::vector<Eigen::Vector3d> _bearings2;
   std::vector<Eigen::Vector2d> _velocities1;
   std::vector<Eigen::Vector2d> _velocities2;
   Eigen::Matrix3d jacobina_q_bg;
   Eigen::Quaterniond qjk_imu;
};

// 3×3 对称半正定矩阵最小特征值的闭式解（值版本，供自动微分使用）。
// 特征多项式: λ^3 - a λ^2 + b λ - c = 0，令 μ = λ - a/3 得 μ^3 + p μ + q = 0，
// 其中 p = b - a²/3 ≤ 0（对称矩阵恒成立），q = c - ab/3 + 2a³/27。
// 三角解: μ_k = 2√(-p/3)·cos(θ/3 - 2πk/3)，θ = acos(R)，
//         R = (3Q/(2P))·√(-3/P)，P=p, Q=-q。
// 最小特征值对应 k=2。
template <typename T>
static T SmallestEigenvalueSym3x3(const Eigen::Matrix<T, 3, 3> &M)
{
    const T a = M(0, 0) + M(1, 1) + M(2, 2);
    const T b = M(0, 0) * M(1, 1) + M(0, 0) * M(2, 2) + M(1, 1) * M(2, 2)
              - M(0, 1) * M(0, 1) - M(0, 2) * M(0, 2) - M(1, 2) * M(1, 2);
    const T c = M(0, 0) * (M(1, 1) * M(2, 2) - M(1, 2) * M(1, 2))
              - M(0, 1) * (M(0, 1) * M(2, 2) - M(1, 2) * M(0, 2))
              + M(0, 2) * (M(0, 1) * M(1, 2) - M(1, 1) * M(0, 2));

    const T p = b - a * a / T(3.0);
    const T q = c - a * b / T(3.0) + T(2.0) * a * a * a / T(27.0);

    if (p > T(-1e-12)) {
        return a / T(3.0);
    }

    const T sqrt_m_p3 = ceres::sqrt(-p / T(3.0));           // sqrt(-p/3) > 0
    const T sqrt_neg3_over_p = ceres::sqrt(T(-3.0) / p);    // sqrt(-3/p) > 0
    T R = (-T(3.0) * q / (T(2.0) * p)) * sqrt_neg3_over_p;  // acos 幅角
    // 数值截断到 [-1, 1]，防止浮点误差使 acos 越界
    R = std::max(T(-1.0), std::min(T(1.0), R));

    const T theta = ceres::acos(R);
    const T phi   = theta / T(3.0);
    // 4π/3 (k=2 对应最小特征值)
    const T kFourPiOverThree = T(4.1887902047863905);
    const T mu_min = T(2.0) * sqrt_m_p3 * ceres::cos(phi - kFourPiOverThree);
    return a / T(3.0) + mu_min;
}

// 文档模型代价函数：单帧对残差 e = sqrt(max(λmin(M), ελ))
class DocRotationCostFunctor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DocRotationCostFunctor(const std::vector<Eigen::Vector3d> &bearings1,
                           const std::vector<Eigen::Vector3d> &bearings2,
                           const Eigen::Quaterniond &q_body,       // 预积分相对旋转 R_bi_bj (delta_q)
                           const Eigen::Matrix3d &J_bg,            // dq_dbg: jacobian.block<3,3>(O_R,O_BG)
                           const Eigen::Vector3d &bg_linearized,   // 积分时使用的陀螺偏置 (linearized_bg)
                           const Eigen::Vector3d &omega_start_raw, // 起点边界原始角速度 (pre_integration->linearized_gyr，区间首样本)
                           const Eigen::Vector3d &omega_end_raw,   // 终点边界原始角速度 (pre_integration->gyr_1，区间末样本)
                           double eps_lambda = 1e-20)
        : _bearings1(bearings1), _bearings2(bearings2),
          _q_body(q_body), _J_bg(J_bg), _bg_linearized(bg_linearized),
          _omega_start_raw(omega_start_raw), _omega_end_raw(omega_end_raw),
          _eps_lambda(eps_lambda) {}

    template <typename T>
    bool operator()(const T *const bg_ptr, const T *const q_bc_ptr, const T *const td_ptr, T *residual) const
    {
        const T td = td_ptr[0];
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg(bg_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q_bc(q_bc_ptr);

        const Eigen::Matrix<T, 3, 1> dbg = bg - _bg_linearized.cast<T>();
        const Eigen::Matrix<T, 3, 1> rot_vec_bg = _J_bg.cast<T>() * dbg;
        Eigen::Matrix<T, 4, 1> q_corr_tmp;
        ceres::AngleAxisToQuaternion(rot_vec_bg.data(), q_corr_tmp.data());
        const Eigen::Quaternion<T> q_bg_corr(q_corr_tmp(0), q_corr_tmp(1), q_corr_tmp(2), q_corr_tmp(3));

        //  --- 2. 时间偏移端点补偿（文档 eq.5/6/11/12）:
        //        R_bi_bj = Exp(ω̃_s·t_d) · R_body · Exp(ω̃_e·t_d)^T,  ω̃ = ω_raw - bg ---
        const Eigen::Matrix<T, 3, 1> w_start = _omega_start_raw.cast<T>() - bg;
        const Eigen::Matrix<T, 3, 1> w_end   = _omega_end_raw.cast<T>() - bg;
        const Eigen::Matrix<T, 3, 1> aa_s = w_start * td;
        const Eigen::Matrix<T, 3, 1> aa_e = w_end * td;
        Eigen::Matrix<T, 4, 1> q_comp_s_tmp, q_comp_e_tmp;
        ceres::AngleAxisToQuaternion(aa_s.data(), q_comp_s_tmp.data());
        ceres::AngleAxisToQuaternion(aa_e.data(), q_comp_e_tmp.data());
        const Eigen::Quaternion<T> q_comp_s(q_comp_s_tmp(0), q_comp_s_tmp(1), q_comp_s_tmp(2), q_comp_s_tmp(3));
        const Eigen::Quaternion<T> q_comp_e(q_comp_e_tmp(0), q_comp_e_tmp(1), q_comp_e_tmp(2), q_comp_e_tmp(3));

        const Eigen::Quaternion<T> q_bij = q_comp_s * (_q_body.cast<T>() * q_bg_corr) * q_comp_e.inverse();

        // --- 3. 预测的相机间相对旋转（文档 eq.10/12）: R_cicj = R_bc^T · R_bij · R_bc ---
        const Eigen::Quaternion<T> q_cicj = q_bc.inverse() * q_bij * q_bc;

        // --- 4. 极平面法向量累加（文档 eq.9）: M = Σ n_k n_k^T, n_k = f_i^k × (R_cicj·f_j^k) ---
        Eigen::Matrix<T, 3, 3> M = Eigen::Matrix<T, 3, 3>::Zero();
        for (int i = 0; i < static_cast<int>(_bearings1.size()); i++) {
            Eigen::Matrix<T, 3, 1> fi = _bearings1[i].cast<T>();
            Eigen::Matrix<T, 3, 1> fj = _bearings2[i].cast<T>();
            fi.normalize();
            fj.normalize();
            const Eigen::Matrix<T, 3, 1> rfj = q_cicj * fj;
            const Eigen::Matrix<T, 3, 1> n = fi.cross(rfj);
            M = M + n * n.transpose();
        }

        // --- 5. 最小特征值残差（文档 eq.26）: e = sqrt(max(λmin, ελ)) ---
        const T lam_min = SmallestEigenvalueSym3x3(M);
        const T e2 = std::max(lam_min, T(_eps_lambda));
        residual[0] = ceres::sqrt(e2);

        return true;
    }

    static ceres::CostFunction *
    Create(const std::vector<Eigen::Vector3d> &bearings1,
           const std::vector<Eigen::Vector3d> &bearings2,
           const Eigen::Quaterniond &q_body,
           const Eigen::Matrix3d &J_bg,
           const Eigen::Vector3d &bg_linearized,
           const Eigen::Vector3d &omega_start_raw,
           const Eigen::Vector3d &omega_end_raw,
           double eps_lambda = 1e-20)
    {
        return (new ceres::AutoDiffCostFunction<DocRotationCostFunctor, 1, 3, 4, 1>(
                new DocRotationCostFunctor(bearings1, bearings2, q_body, J_bg, bg_linearized,
                                           omega_start_raw, omega_end_raw, eps_lambda)));
    }

private:
    std::vector<Eigen::Vector3d> _bearings1;
    std::vector<Eigen::Vector3d> _bearings2;
    Eigen::Quaterniond _q_body;
    Eigen::Matrix3d _J_bg;
    Eigen::Vector3d _bg_linearized;
    Eigen::Vector3d _omega_start_raw;
    Eigen::Vector3d _omega_end_raw;
    double _eps_lambda;
};
