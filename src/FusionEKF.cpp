#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1,0,0,0,
            0,1,0,0;

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;



}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
        TODO:
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // first measurement
        cout << "EKF: " << endl;
        VectorXd x(4);
        MatrixXd P(4, 4);



        P = MatrixXd(4, 4);
        P << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;



        double px = 0;
        double py = 0;

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rho_dot = measurement_pack.raw_measurements_[2];
            px = rho * cos(phi);
            py = rho * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);
            x << px, py, vx, vy;
        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            px = measurement_pack.raw_measurements_[0];
            py = measurement_pack.raw_measurements_[1];
            x << px, py, 0, 0;
        }
        if (fabs(x(0)) < 0.0001 and fabs(x(1)) < 0.0001) {
            x(0) = 0.0001;
            x(1) = 0.0001;
        }

        ekf_.Init(x, P);
        previous_timestamp_ = measurement_pack.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/


    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;    //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    //ekf_.F_(0, 2) = dt;
    //ekf_.F_(1, 3) = dt;
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1;


    float noise_ax = 9.0;
    float noise_ay = 9.0;

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    float dt3_div2 = dt3 / 2;
    float dt4_div4 = dt4 / 4;

    //2. Set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt4_div4 * noise_ax, 0, dt3_div2 * noise_ax, 0,
            0, dt4_div4 * noise_ay, 0, dt3_div2 * noise_ay,
            dt3_div2 * noise_ax, 0, dt2 * noise_ax, 0,
            0, dt3_div2 * noise_ay, 0, dt2 * noise_ay;


    ekf_.Predict();

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     TODO:
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);

    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

    // print the output
    cout << "x_ = " << ekf_.x_ << endl;
    cout << "P_ = " << ekf_.P_ << endl;
}
