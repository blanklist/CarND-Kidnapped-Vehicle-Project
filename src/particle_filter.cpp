/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  // Set the number of particles

  num_particles = 100;  

  // normal distributions
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize all particles to first position with added noise
  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;

    particles.push_back(p);
  }

  is_initialized = true;
  
  cout << "INITIALIZATION complete\n";
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // normal distributions
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {

    // add measurement to each particle
    if (fabs(yaw_rate) < 0.00001) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
  cout << "PREDICTION complete\n";
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  // current observation
  for (LandmarkObs& o: observations) {
    // set minimum distance to maximum possible
    double min_dist = numeric_limits<double>::max();
    // current prediction
    for (LandmarkObs& p: predicted){
      // distance between observed and predicted landmarks
      double current_dist = dist(o.x, o.y, p.x, p.y);
      // find closest predicted nearest current observed
      if (current_dist < min_dist) {
        // reset
        min_dist = current_dist;
        o.id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // par  = particles
  // obvs = observation
  // pre  = predictions

  for (int i = 0; i < num_particles; i++) {

    double par_x =     particles[i].x;
    double par_y =     particles[i].y;
    double par_theta = particles[i].theta;


  // select the landmarks in range
    vector<LandmarkObs> predictions;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++){
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int   landmark_id = map_landmarks.landmark_list[j].id_i;

      double distance = dist(par_x, par_y, landmark_x, landmark_y);
      if (fabs(landmark_x - par_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }

  // transform observations from VEH to MAP coordinates
    vector<LandmarkObs> trans_obvs;
    for (unsigned int j = 0; j < observations.size(); j++) {
      double trans_x = cos(par_theta) * observations[j].x - sin(par_theta) * observations[j].y + par_x;
      double trans_y = sin(par_theta) * observations[j].x + cos(par_theta) * observations[j].y + par_y;
      double trans_id = observations[j].id;
      trans_obvs.push_back(LandmarkObs{trans_id, trans_x, trans_y});
    }

  // associate observation with closest landmark
    dataAssociation(predictions, trans_obvs);

    particles[i].weight = 1.0;

  // update particle wieght based on observed distance and actual position of landmark
    for (unsigned int j = 0; j < trans_obvs.size(); j++) {
      double obvs_x, obvs_y, pre_x, pre_y;
      obvs_x = trans_obvs[j].x;
      obvs_y = trans_obvs[j].y;

      // coordinates of assiciate prediction
      for (int k = 0; k < predictions.size(); k++) {
        if (predictions[k].id == trans_obvs[j].id) {
          pre_x = predictions[k].x;
          pre_y = predictions[k].y;
        }
      }

      double x_weight = std_landmark[0];
      double y_weight = std_landmark[1];
      double weight = (1 / (2 * M_PI * x_weight * y_weight)) * exp( -(pow(pre_x - obvs_x, 2) / (2 * pow(x_weight, 2)) + (pow(pre_y - obvs_y, 2) / (2 * pow(y_weight, 2)))));

      particles[i].weight *= weight;
//       if (particles[i].weight != 0){
// 	    cout << particles[i].weight;
//       }
    }
  }
  cout << "WEIGHTS UPDATED\n";
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // // generate distribution
  // random_device rd;
  // mt19937 gen(rd());
  // discrete_distribution<> dist(weights.begin(), weights.end());

  // // resampled particles init
  // vector<Particle> resampled_particles;
  // resampled_particles.resize(num_particles);

  // // resample according to weights
  // for (int i = 0; i < num_particles; i++) {
  //   int j = dist(gen);
  //   resampled_particles[i] = particles[j];
  // }

  // particles = resampled_particles;

  // weights.clear();
  // cout << "RESAMPLE complete\n";
  vector<double> weights;
  double max_weight = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
  }

  // uniform real distribution
  // urd<double> double_dist(0.0, max_weight);
  urd<double> double_dist(0.0, max_weight);
  //uniform integer distribution
  uid<int> integer_dist(0, num_particles - 1);
  int index = dist_integer(gen);
  double beta = 0.0
  vector<Particle> = resampled_particles;
  for (int i = 0; i < num_particles; i++) {
    beta += double_dist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}