#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  I implemented this from https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/3612b91d-9c33-47ad-8067-a572a6c93837/concepts/a0604e14-2832-4646-835a-05f972453f3d
  u is 0, 0 because the noise mean is 0
  */
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  UpdateFromY(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    double rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
    double theta = atan(x_(1) / x_(0));
    double rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;

    VectorXd h = VectorXd(3);
    h << rho, theta, rho_dot;

    VectorXd y = z - h;

    //Normalize angles to between -pi and pi
    if (y(1) < -M_PI){
        while(y(1) < -M_PI){
            y(1) += (2 * M_PI);
        }
    }
    else if(y(1) > M_PI){
        while(y(1) < M_PI){
            y(1) -= (2 * M_PI);
        }
    }

    UpdateFromY(y);
}

void KalmanFilter::UpdateFromY(const VectorXd &y){
    MatrixXd Ht = H_.transpose();
    MatrixXd S = Ht * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    // update state vector and covariance matrix
    x_ = x_ + (K * y);
    MatrixXd I = MatrixXd::Identity(2, 2);
    P_ = (I - K * H_) * P_;
}
