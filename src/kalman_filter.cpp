#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in) {
  x_ = x_in;
  P_ = P_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd y = z - H_ * x_;
    UpdateFromY(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

    float px = x_(0);
    float py = x_(1);
    float vx = x_(2);
    float vy = x_(3);
    float rho = sqrt(pow(px,2)+pow(py,2));
    float theta = atan2(py,px);
    float rho_dot = (px * vx + py * vy)/rho;

    VectorXd h = VectorXd(3);
    h << rho, theta, rho_dot;

    VectorXd y = z - h;

    if( y(1) > PI_ )
        y(1) -= 2*PI_;
    if( y(1) < -PI_ )
        y(1) += 2*PI_;

    UpdateFromY(y);
}

void KalmanFilter::UpdateFromY(const VectorXd &y){
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    // update state vector and covariance matrix
    x_ = x_ + (K * y);
    MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());

    P_ = (I - K * H_) * P_;
}
