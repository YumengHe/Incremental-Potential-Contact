#include "jello.h"
#include "physics.h"
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <algorithm>

// Eigen
#include "eigen-3.4.0/Eigen/Dense"
#include "eigen-3.4.0/Eigen/Sparse"
#include "eigen-3.4.0/Eigen/SparseCholesky"

using namespace std;
using namespace Eigen;

struct point gravity = {0.0, 0.0, -9.81};
double dhat = 0.01;
double kappa = 1e5;
double BMIN = -2.0;
double BMAX =  2.0;


// math operations
struct point vec_minus_vec(struct point a, struct point b) {
  struct point result;
  result.x = a.x - b.x;
  result.y = a.y - b.y;
  result.z = a.z - b.z;
  return result;
}

struct point vec_plus_vec(struct point a, struct point b) {
  struct point result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

struct point vec_time_scale(struct point a, double b) {
  struct point result;
  result.x = a.x * b;
  result.y = a.y * b;
  result.z = a.z * b;
  return result;
}

double vec_dot_vec(struct point a, struct point b) {
  double result;
  result = a.x * b.x + a.y * b.y + a.z * b.z;
  return result;
}

// find index
int nodeIndex(int i, int j, int k) {
  return (i * num_points_per_line * num_points_per_line) + (j * num_points_per_line) + k;
}

int indexI(int x) {
  return int(x / (num_points_per_line * num_points_per_line));
}

int indexJ(int x) {
  return int((x / num_points_per_line) % num_points_per_line);
}

int indexK(int x) {
  return int(x % num_points_per_line);
}

// helper functions
void copyJelloPositions(world *jello, vector<point> &x) {
  x.resize(num_points);
  for(int i = 0; i < num_points_per_line; i++){
    for(int j = 0; j < num_points_per_line; j++){
      for(int k = 0; k < num_points_per_line; k++){
        x[nodeIndex(i,j,k)] = jello->p[i][j][k];
      }
    }
  }
}

// compute edges
vector< array<int, 2> > generateJelloEdges() {
  vector< array<int, 2> > edges;
  edges.reserve(6000);

  // ------ structural edges ------
  for(int i = 0; i < num_points_per_line; i++) {
    for(int j = 0; j < num_points_per_line; j++) {
      for(int k = 0; k < num_points_per_line; k++) {
        int idx = nodeIndex(i, j, k);
        // along x
        if(i + 1 < num_points_per_line) {
          array<int,2> edgeTmp = {idx, nodeIndex(i+1, j, k)};
          edges.push_back(edgeTmp);
        }
        // along y
        if(j + 1 < num_points_per_line) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j+1, k)};
          edges.push_back(edgeTmp);
        }
        // along z
        if(k + 1 < num_points_per_line) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j, k+1)};
          edges.push_back(edgeTmp);
        }
      }
    }
  }

  // ------ shear edges ------
  int shearOffsets[20][3] = {
    {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1,-1, 0},
    {0, 1, 1}, {0, 1,-1}, {0,-1, 1}, {0, -1, -1},
    {1, 0, 1}, {1, 0,-1}, {-1,0, 1}, {-1,0,-1},
    {1, 1, 1}, {1, 1,-1}, {1,-1, 1}, {-1,1, 1}, { -1,-1,1}, { -1,1,-1}, { 1,-1,-1}, { -1,-1,-1}
  };

  for(int i = 0; i < num_points_per_line; i++) {
    for(int j = 0; j < num_points_per_line; j++) {
      for(int k = 0; k < num_points_per_line; k++) {
        int idx = nodeIndex(i, j, k);
        // Try each offset in the shearOffsets list
        for(auto &ofs : shearOffsets) {
            int ni = i + ofs[0]; // neighbor i
            int nj = j + ofs[1]; // neighbor j
            int nk = k + ofs[2]; // neighbor k

          // check boundaries
          if(ni >= 0 && ni < num_points_per_line &&
              nj >= 0 && nj < num_points_per_line &&
              nk >= 0 && nk < num_points_per_line) {
            // To avoid duplicating edges in both directions,
            // only push if (ni, nj, nk) > (i, j, k) in flattened order.
            // That is, we only add edge if new_idx > idx.
            int new_idx = nodeIndex(ni, nj, nk);
            if(new_idx > idx)
            {
              array<int,2> edgeTmp = {idx, new_idx};
              edges.push_back(edgeTmp);
            }
          }
        }
      }
    }
  }

  // ------ bend edges ------
  for(int i = 0; i < num_points_per_line; i++) {
    for(int j = 0; j < num_points_per_line; j++) {
      for(int k = 0; k < num_points_per_line; k++) {
        int idx = nodeIndex(i, j, k);
        // i+2
        if(i + 2 < num_points_per_line) {
          array<int,2> edgeTmp = {idx, nodeIndex(i+2, j, k)};
          edges.push_back(edgeTmp);
        }
        // j+2
        if(j + 2 < num_points_per_line) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j+2, k)};
          edges.push_back(edgeTmp);
        }
        // k+2
        if(k + 2 < num_points_per_line) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j, k+2)};
          edges.push_back(edgeTmp);
        }
      }
    }
  }

  return edges;
}

// --------------------------------------------------
// Inertia Term
// --------------------------------------------------
double valInertia(vector<Vector3d> &x, vector<Vector3d> &x_tilde, double m) {
  double sum = 0.0;
  for (int i = 0; i < num_points; i ++) {
    Vector3d diff = x[i] - x_tilde[i];
    sum += 0.5 * m * diff.dot(diff);
  }
  return sum;
}

vector<Vector3d> gradInertia(vector<Vector3d> &x, vector<Vector3d> &x_tilde, double m) {
  vector<Vector3d> g(num_points);
  for (int i = 0; i < num_points; i ++) {
    g[i] = m * (x[i] - x_tilde[i]);
  }
  return g;
}

Matrix3d hessInertia(vector<Vector3d> &x, vector<Vector3d> &xTilde, double m) {
  Matrix3d IJV(num_points * 3);
  for (int i = 0; i < num_points; i ++) {
    for (int d = 0; d < 3; d ++) {
      IJV[0, i * 3 + d] = i * 3 + d;
      IJV[1, i * 3 + d] = i * 3 + d;
      IJV[2, i * 3 + d] = m;
    }
  }
  return IJV;
}

// --------------------------------------------------
// Mass-Spring Potential Energy
// --------------------------------------------------
// l2: length of the spring
double valSpring(vector<Vector3d> &x, vector< array<int,2> > &e, vector<double> &l2, double k) {
  double sum = 0.0;
  for(int i = 0; i < e.size(); i++) {
    Vector3d diff = x[e[i][0]] - x[e[i][1]];
    sum += l2[i] * 0.5 * k * pow((diff.dot(diff) / l2[i] - 1), 2);
  }
  return sum;
}

vector<Vector3d> gradSpring(vector<Vector3d> &x, vector< array<int,2> > &e, vector<double> &l2, double k) {
  vector<Vector3d> g(num_points);
    
  // initialize
  for (int i = 0; i < num_points; i++){
    g[i] = Vector3d(0.0, 0.0, 0.0);
  }

  for(int i = 0; i < e.size(); i++) {
    Vector3d diff = x[e[i][0]] - x[e[i][1]];
    Vector3d g_diff = 2.0 * k * (diff.dot(diff) / l2[i] - 1) * diff;
    g[e[i][0]] += g_diff;
    g[e[i][1]] -= g_diff;
  }

  return g;
}

MatrixXd block(Matrix3d &H_diff) {
  // 6 * 6 matrix
  // [[ Hdiff, -Hdiff ],
  //  [ -Hdiff, Hdiff ]]
  MatrixXd result(6, 6);
  result.block<3,3>(0,0) = H_diff;
  result.block<3,3>(0,3) = -H_diff;
  result.block<3,3>(3,0) = -H_diff;
  result.block<3,3>(3,3) = H_diff;
  return result;
}

MatrixXd make_PSD(MatrixXd hess) {
  // Eigen decomposition on symmetric matrix
  SelfAdjointEigenSolver<MatrixXd> eigensolver(hess);
  VectorXd lam = eigensolver.eigenvalues();
  MatrixXd V = eigensolver.eigenvectors();

  // Set all negative Eigenvalues to 0
  for (int i = 0; i < lam.size(); i ++) {
    lam[i] = std::max(0.0, lam[i]);
  }

  // np.diag(lam)
  MatrixXd D = lam.asDiagonal();

  return V * D * V.transpose();
}

MatrixXd hessSpring(vector<Vector3d> &x, vector< array<int,2> > &e, vector<double> &l2, double k) {
  // Each edge contributes 6x6 = 36 entries to the global Hessian
  MatrixXd IJV(3, e.size() * 36);

  for(int i = 0; i < e.size(); i++) {
    // compute diff
    Vector3d diff = x[e[i][0]] - x[e[i][1]];
    Matrix3d Hdiff = 2 * k / l2[i] * (2 * diff * diff.transpose() + (diff.dot(diff) - l2[i]) * Matrix2d::Identity(2,2));
    MatrixXd H_local = make_PSD(block(Hdiff));
    
    // add to global matrix
    for (int nI = 0; nI < 2; nI ++) {
      for (int nJ = 0; nJ < 2; nJ ++) {
        int indStart = i * 26 + (nI * 3 + nJ) * 9;
        for (int r = 0; r < 3; r ++) {
          for (int c = 0; c < 3; c ++) {
            IJV[0, indStart + r * 3 + c] = e[i][nI] * 3 + r;
            IJV[1, indStart + r * 3 + c] = e[i][nJ] * 3 + r;
            IJV[2, indStart + r * 3 + c] = H_local[nI * 3 + r, nJ * 3 + c];
          }
        }
      }
    }
  }
  return IJV;
}

// Gravity Energy
// double valGravity(world *jello, double m) {
//   double sum = 0.0;
//   for (int i = 0; i < num_points; i++) {
//     sum += -m * vec_dot_vec(jello->p[indexI(i)][indexJ(i)][indexK(i)], gravity);
//   }
//   return sum;
// }

// double temp_valGravity(vector<point> &x, double m) {
//   double sum = 0.0;
//   for (int i = 0; i < num_points; i++) {
//     sum += -m * vec_dot_vec(x[i], gravity);
//   }
//   return sum;
// }

// vector<point> gradGravity(world *jello, double m) {
//   vector<point> g(num_points);
//   for (int i = 0; i < num_points; i++) {
//     g[i].x = -m * gravity.x;
//     g[i].y = -m * gravity.y;
//     g[i].z = -m * gravity.z;
//   }
//   return g;
// }

// Barrier Energy
// double barrierVal(world *jello, double z_ground, double contact_area) {
//   double sumVal = 0.0;
//   for(size_t i=0; i < num_points; i++)
//   { 
//     double px = jello->p[indexI(i)][indexJ(i)][indexK(i)].x;
//     double py = jello->p[indexI(i)][indexJ(i)][indexK(i)].y;
//     double pz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
//     double d[6];
//     d[0] = (BMAX - px);
//     d[1] = (px - BMIN);
//     d[2] = (BMAX - py);
//     d[3] = (py - BMIN);
//     d[4] = (BMAX - pz);
//     d[5] = (pz - BMIN);
//     for(int k=0; k<6; k++) {
//       if(d[k] < dhat)
//       {
//           double s = d[k] / dhat;
//           sumVal += contact_area * dhat * (0.5*kappa) * (s - 1.0)*log(s);
//       }
//     }
//   }
//   return sumVal;
// }

double barrierVal(world *jello, double z_ground, double contact_area) {
  double sum = 0.0;
  for(int i=0; i < num_points; i++) { 
    double px = jello->p[indexI(i)][indexJ(i)][indexK(i)].x;
    double py = jello->p[indexI(i)][indexJ(i)][indexK(i)].y;
    double pz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
    double d = pz - z_ground;
    if(d < dhat) {
      double s = d / dhat;
      sum += contact_area * dhat * kappa / 2.0 * (s - 1.0) * log(s);
    }
  }
  return sum;
}

// double temp_barrierVal(vector<point> &x, double z_ground, double contact_area) {
//   double sumVal = 0.0;
//   for(int i=0; i < num_points; i++)
//   { 
//     double px = x[i].x;
//     double py = x[i].y;
//     double pz = x[i].z;
//     double d[6];
//     d[0] = (BMAX - px);
//     d[1] = (px - BMIN);
//     d[2] = (BMAX - py);
//     d[3] = (py - BMIN);
//     d[4] = (BMAX - pz);
//     d[5] = (pz - BMIN);
//     for(int k=0; k<6; k++) {
//       if(d[k] < dhat)
//       {
//           double s = d[k] / dhat;
//           sumVal += contact_area * dhat * (0.5*kappa) * (s - 1.0)*log(s);
//       }
//     }
//   }
//   return sumVal;
// }

double temp_barrierVal(vector<point> &x, double z_ground, double contact_area) {
  double sum = 0.0;
  for(int i=0; i < num_points; i++) { 
    double px = x[i].x;
    double py = x[i].y;
    double pz = x[i].z;
    double d = pz - z_ground;
    if(d < dhat) {
      double s = d / dhat;
      sum += contact_area * dhat * kappa / 2.0 * (s - 1.0) * log(s);
    }
  }
  return sum;
}

// vector<point> barrierGrad(world *jello, double z_ground, double contact_area) {
//   vector<point> g(num_points);
//   for(size_t i=0; i<num_points; i++){
//       g[i].x = 0.0; g[i].y = 0.0; g[i].z = 0.0;
//   }
//
//   for(size_t i=0; i < num_points; i++)
//   { 
//     double px = jello->p[indexI(i)][indexJ(i)][indexK(i)].x;
//     double py = jello->p[indexI(i)][indexJ(i)][indexK(i)].y;
//     double pz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
//     double d[6];
//     // plane x=+2 => d[0], normal is -x
//     d[0] = BMAX - px;
//     // plane x=-2 => d[1], normal is +x
//     d[1] = px - BMIN;
//
//     // plane y=+2 => d[2], normal is -y
//     d[2] = BMAX - py;
//     // plane y=-2 => d[3], normal is +y
//     d[3] = py - BMIN;
//
//     // plane z=+2 => d[4], normal is -z
//     d[4] = BMAX - pz;
//     // plane z=-2 => d[5], normal is +z
//     d[5] = pz - BMIN;
//     for(int k=0; k<6; k++)
//         {
//             if(d[k] < dhat)
//             {
//                 double s = d[k]/dhat;
//                 // from the 2D code: 
//                 // grad[i][normal_dir] = contact_area[i]*dhat*(kappa/2)*( log(s)/dhat + (s-1)/d )
//                 // We must figure out the sign to get the correct direction.
//
//                 // Let's define the partial formula as:
//                 double val = contact_area*dhat*(0.5*kappa)*( std::log(s)/dhat + (s-1.0)/d[k] );
//
//                 // Then apply it in the direction of the plane's normal
//                 // plane 0 => normal = -X => g[i].x -= val
//                 // plane 1 => normal = +X => g[i].x += val
//                 // plane 2 => normal = -Y => g[i].y -= val
//                 // plane 3 => normal = +Y => g[i].y += val
//                 // plane 4 => normal = -Z => g[i].z -= val
//                 // plane 5 => normal = +Z => g[i].z += val
//                 switch(k)
//                 {
//                     case 0: g[i].x -= val; break; // x=+2
//                     case 1: g[i].x += val; break; // x=-2
//                     case 2: g[i].y -= val; break; // y=+2
//                     case 3: g[i].y += val; break; // y=-2
//                     case 4: g[i].z -= val; break; // z=+2
//                     case 5: g[i].z += val; break; // z=-2
//                 }
//             }
//         }
//   }
//   return g;
// }

vector<point> barrierGrad(world *jello, double z_ground, double contact_area) {
  vector<point> g(num_points);
  for(int i=0; i<num_points; i++){
    g[i].x = 0.0; g[i].y = 0.0; g[i].z = 0.0;
  }

  for(int i=0; i < num_points; i++) { 
    double px = jello->p[indexI(i)][indexJ(i)][indexK(i)].x;
    double py = jello->p[indexI(i)][indexJ(i)][indexK(i)].y;
    double pz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
    double d = pz - z_ground;
    if(d < dhat) {
      double s = d / dhat;
      double val = contact_area * dhat * (kappa / 2.0 * (log(s) / dhat + (s - 1.0) / d));
      g[i].z = val;
    }
  }
  return g;
}

// IJV barrierHess(world *jello, double z_ground, double contact_area) {
//   IJV out;
//   out.I.resize(num_points);
//   out.J.resize(num_points);
//   out.V.resize(num_points);
//
//   for(int i=0; i < num_points; i++)
//   { 
//     double px = jello->p[indexI(i)][indexJ(i)][indexK(i)].x;
//         double py = jello->p[indexI(i)][indexJ(i)][indexK(i)].y;
//         double pz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
//
//         double d[6];
//         d[0] = BMAX - px; // plane x=+2 => normal -X
//         d[1] = px - BMIN; // plane x=-2 => normal +X
//         d[2] = BMAX - py; // plane y=+2 => normal -Y
//         d[3] = py - BMIN; // plane y=-2 => normal +Y
//         d[4] = BMAX - pz; // plane z=+2 => normal -Z
//         d[5] = pz - BMIN; // plane z=-2 => normal +Z
//
//         for(int k=0; k<6; k++)
//         {
//             if(d[k]<dhat)
//             {
//                 // from 2D code: H(i) = contact_area[i]*dhat*kappa / (2*d*d*dhat)*(d + dhat)
//                 // We'll store it in the correct dof (3*i+0, or +1, or +2).
//                 double val = contact_area*dhat*kappa/(2.0*d[k]*d[k]*dhat)*(d[k]+dhat);
//
//                 // which dof?
//                 int dof = -1;
//                 // plane 0 => x dimension => dof=3*i+0
//                 // plane 1 => x dimension => dof=3*i+0
//                 // plane 2 => y dimension => dof=3*i+1
//                 // plane 3 => y dimension => dof=3*i+1
//                 // plane 4 => z dimension => dof=3*i+2
//                 // plane 5 => z dimension => dof=3*i+2
//                 int coord = 0; // x=0, y=1, z=2
//                 if(k==0 || k==1) coord=0;
//                 else if(k==2 || k==3) coord=1;
//                 else if(k==4 || k==5) coord=2;
//
//                 int globalRow = 3*static_cast<int>(i)+coord;
//                 int globalCol = globalRow;
//
//                 // We place it as a diagonal entry (globalRow, globalCol)
//                 out.I.push_back(globalRow);
//                 out.J.push_back(globalCol);
//                 out.V.push_back(val);
//             }
//         }
//   }
//
//   return out;
// }

IJV barrierHess(world *jello, double z_ground, double contact_area) {
  IJV out;
  out.I.resize(num_points);
  out.J.resize(num_points);
  out.V.resize(num_points);

  for(int i=0; i < num_points; i++) { 
    out.I[i] = i * 3 + 2;
    out.J[i] = i * 3 + 2;

    double px = jello->p[indexI(i)][indexJ(i)][indexK(i)].x;
    double py = jello->p[indexI(i)][indexJ(i)][indexK(i)].y;
    double pz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
    double d = pz - z_ground;
    if(d < dhat) {
      double val = contact_area * dhat * kappa / (2.0 * d * d * dhat) * (d + dhat);
      out.V[i] = val;
    } else {
      out.V[i] = 0.0;
    }
  }
  return out;
}

// double init_step_size(world *jello, double z_ground, vector<point> &p) {
//   double alpha = 1.0;
//   for(int i=0; i<num_points; i++) {
//     double pz = p[i].z;
//     if(pz < 0.0 && jello->p[indexI(i)][indexJ(i)][indexK(i)].y > z_ground)
//     {
//       double candidate = 0.9 * (z_ground - jello->p[indexI(i)][indexJ(i)][indexK(i)].y) / py;
//       if(candidate < alpha && candidate > 0.0)
//         alpha = candidate;
//     }
//   }
//   return alpha;
// }

double init_step_size(world *jello, double z_ground, vector<point> &p) {
  double alpha = 1.0;
  for(int i=0; i<num_points; i++) {
    double pz = p[i].z;
    double curz = jello->p[indexI(i)][indexJ(i)][indexK(i)].z;
    if(curz - z_ground > 1e-6 && pz < 0.0) {
      double candidate = 0.9 * (z_ground - curz) / pz;
      if(candidate < alpha && candidate > 0.0) {
        alpha = candidate;
      }
    }
  }
  return alpha;
}

// --------------------------------------------------
// Optimization Time Integrator
// --------------------------------------------------
double IP_val(vector<Vector3d> &x, vector< array<int,2> > &e, vector<Vector3d> &x_tilde, double m, vector<double> &l2, double k, double h, double z_ground, double contact_area) {
  double result;
  double Ei = valInertia(x, x_tilde, m);
  double Es = valSpring(x, e, l2, k);
  // double Eg = valGravity(x, m);
  double Eb = barrierVal(x, z_ground, contact_area);
  
  result = Ei + h * h * (Es + Eb);

  if (grav == 1) {
    result += (h * h * Eg);
  }
  return result;
}

vector<point> IP_grad(world *jello, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h, double z_ground, double contact_area) {
  vector<point> gI = gradInertia(jello, x_tilde, m);
  vector<point> gS = gradSpring(jello, e, l2, k);
  vector<point> gB = barrierGrad(jello, z_ground, contact_area);
  
  for(int i=0; i<num_points; i++) {
    gI[i].x += (h * h * (gS[i].x + gB[i].x));
    gI[i].y += (h * h * (gS[i].y + gB[i].y));
    gI[i].z += (h * h * (gS[i].z + gB[i].z));
  }

  if (grav == 1) {
    vector<point> gG = gradGravity(jello, m);
    for(int i=0; i<num_points; i++) {
      gI[i].x += (h * h * gG[i].x);
      gI[i].y += (h * h * gG[i].y);
      gI[i].z += (h * h *gG[i].z);
    }
  }

  return gI;
}

IJV IP_hess(world *jello, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h, double z_ground, double contact_area) {
  IJV Hi = hessInertia(jello, x_tilde, m);
  IJV Hs = hessSpring(jello, e, l2, k);
  IJV Hb = barrierHess(jello, z_ground, contact_area);

  for(int i=0; i < Hs.V.size(); i++){
    Hs.V[i] *= (h * h);
  }
  for(int i=0; i < Hb.V.size(); i++){
    Hb.V[i] *= (h * h);
  }
  
  // combine Hi and hSp
  IJV combined;
  combined.I.reserve(Hi.I.size() + Hs.I.size() + Hb.I.size());
  combined.J.reserve(Hi.J.size() + Hs.J.size() + Hb.J.size());
  combined.V.reserve(Hi.V.size() + Hs.V.size() + Hb.V.size());

  // inertia parts
  combined.I.insert(combined.I.end(), Hi.I.begin(), Hi.I.end());
  combined.J.insert(combined.J.end(), Hi.J.begin(), Hi.J.end());
  combined.V.insert(combined.V.end(), Hi.V.begin(), Hi.V.end());

  // spring parts
  combined.I.insert(combined.I.end(), Hs.I.begin(), Hs.I.end());
  combined.J.insert(combined.J.end(), Hs.J.begin(), Hs.J.end());
  combined.V.insert(combined.V.end(), Hs.V.begin(), Hs.V.end());

  // barrier parts
  combined.I.insert(combined.I.end(), Hb.I.begin(), Hb.I.end());
  combined.J.insert(combined.J.end(), Hb.J.begin(), Hb.J.end());
  combined.V.insert(combined.V.end(), Hb.V.begin(), Hb.V.end());

  return combined;
}

vector<double> solveSparseSystemEigen(IJV &A, vector<double> &rhs, int n)
{
  // convert IJV triplets to Eigen triplets
  vector< Triplet<double> > triplets;
  triplets.reserve(A.I.size());
  for(size_t k = 0; k < A.I.size(); k++) {
    triplets.push_back(Triplet<double>(A.I[k], A.J[k], A.V[k]));
  }

  // 2) Build the sparse matrix
  SparseMatrix<double> mat(n, n);
  mat.setFromTriplets(triplets.begin(), triplets.end());

  // 3) Build the Eigen vector for RHS
  VectorXd b(n);
  for(int i = 0; i < n; i++) {
    b[i] = rhs[i];
  }

  SimplicialLDLT< SparseMatrix<double> > solver;
  solver.compute(mat);
  if(solver.info() != Success) {
    cout << "solve not success";
    return vector<double>(n, 0.0);
  }

  VectorXd x = solver.solve(b);
  if(solver.info() != Success) {
    cout << "solve not success2";
    return std::vector<double>(n, 0.0);
  }

  // 5) Convert x back to std::vector<double>
  vector<double> xSol(n);
  for(int i = 0; i < n; i++) {
    xSol[i] = x[i];
  }
  return xSol;
}

vector<point> search_dir(world *jello, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h, double z_ground, double contact_area) {
  IJV A = IP_hess(jello, x_tilde, m, e, l2, k, h, z_ground, contact_area);
  vector<point> g = IP_grad(jello, x_tilde, m, e, l2, k, h, z_ground, contact_area);

  // Flatten g into -g for the RHS
  int N = num_points;
  vector<double> rhs(3*N);
  for(int i=0; i < N; i++) {
    rhs[3*i + 0] = -g[i].x;
    rhs[3*i + 1] = -g[i].y;
    rhs[3*i + 2] = -g[i].z;
  }

  // Solve the system: A * p = rhs
  vector<double> p_sol = solveSparseSystemEigen(A, rhs, 3*N);
  // Unflatten p_sol into a vector<point>
  vector<point> p(N);
  for(int i=0; i < N; i++){
    p[i].x = p_sol[3*i + 0];
    p[i].y = p_sol[3*i + 1];
    p[i].z = p_sol[3*i + 2];
  }
  return p;
}

// Simulation
double normInfOverH(vector<point> &p, double h) {
  double maxVal = 0.0;
  for(int i = 0; i < p.size(); i++) {
    double absx = fabs(p[i].x);
    double absy = fabs(p[i].y);
    double absz = fabs(p[i].z);
    if (absx > maxVal) {
      maxVal = absx;
    }
    if (absy > maxVal) {
      maxVal = absy;
    }
    if (absz > maxVal) {
      maxVal = absz;
    }
  }
  return maxVal / h;
};

void stepForwardImplicitEuler(world *jello, vector< array<int,2> > &edges, double m, vector<double> &l2, double k, double h, double tol, vector<point> &x_old, double z_ground, double contact_area) {

  // x_tilde
  vector<point> x_tilde(num_points);
  for(size_t i=0; i < num_points; i++) {
    x_tilde[i].x = jello->p[indexI(i)][indexJ(i)][indexK(i)].x + h * jello->v[indexI(i)][indexJ(i)][indexK(i)].x;
    x_tilde[i].y = jello->p[indexI(i)][indexJ(i)][indexK(i)].y + h * jello->v[indexI(i)][indexJ(i)][indexK(i)].y;
    x_tilde[i].z = jello->p[indexI(i)][indexJ(i)][indexK(i)].z + h * jello->v[indexI(i)][indexJ(i)][indexK(i)].z;
  }

  // newton loop
  int iter = 0;
  double E_last = IP_val(jello, x_tilde, m, edges, l2, k, h, z_ground, contact_area);
  // cout << " E_last: " << E_last;
  vector<point> p = search_dir(jello, x_tilde, m, edges, l2, k, h, z_ground, contact_area);
  while(normInfOverH(p, h) > tol && iter < 30) {
    
    // line search
    double alpha = init_step_size(jello, z_ground, p);
    while(true) {
      vector<point> tempX(num_points);
      for(int i=0; i < num_points; i++){
        tempX[i].x = jello->p[indexI(i)][indexJ(i)][indexK(i)].x + alpha * p[i].x;
        tempX[i].y = jello->p[indexI(i)][indexJ(i)][indexK(i)].y + alpha * p[i].y;
        tempX[i].z = jello->p[indexI(i)][indexJ(i)][indexK(i)].z + alpha * p[i].z;
      }
      double E_try = temp_IP_val(tempX, x_tilde, m, edges, l2, k, h, z_ground, contact_area);
      if(E_try <= E_last){
        // accept
        break;
      }
      alpha /= 2.0;
      if(alpha < 1e-12) {
        break;
      }
    }

    // update x
    for(int i=0; i < num_points; i++){
      jello->p[indexI(i)][indexJ(i)][indexK(i)].x += alpha * p[i].x;
      jello->p[indexI(i)][indexJ(i)][indexK(i)].y += alpha * p[i].y;
      jello->p[indexI(i)][indexJ(i)][indexK(i)].z += alpha * p[i].z;
    }
    E_last = IP_val(jello, x_tilde, m, edges, l2, k, h, z_ground, contact_area);
    p = search_dir(jello, x_tilde, m, edges, l2, k, h, z_ground, contact_area);
    iter++;
  }

  // v = (x - x_n) / h
  for(int i=0; i < num_points; i++) {
    jello->v[indexI(i)][indexJ(i)][indexK(i)].x = (jello->p[indexI(i)][indexJ(i)][indexK(i)].x - x_old[i].x) / h;
    jello->v[indexI(i)][indexJ(i)][indexK(i)].y = (jello->p[indexI(i)][indexJ(i)][indexK(i)].y - x_old[i].y) / h;
    jello->v[indexI(i)][indexJ(i)][indexK(i)].z = (jello->p[indexI(i)][indexJ(i)][indexK(i)].z - x_old[i].z) / h;
    // cout << jello->v[indexI(i)][indexJ(i)][indexK(i)].z;
  }

  // write results back to jello->p and jello->v
  // copyVectorsToJello(jello, x, v);
}

void simulate(world *jello) {
  auto edges = generateJelloEdges();
  double tolerance = 1e-2;
  double timeStep = jello->dt;
  double m = jello->mass;
  double k = jello->kElastic;

  vector<point> xFlat;
  copyJelloPositions(jello, xFlat);

  vector<double> restLenSquared(edges.size());
  for (int i = 0; i < edges.size(); i++) {
    int idxA = edges[i][0];
    int idxB = edges[i][1];
    point diff;
    diff.x = xFlat[idxA].x - xFlat[idxB].x;
    diff.y = xFlat[idxA].y - xFlat[idxB].y;
    diff.z = xFlat[idxA].z - xFlat[idxB].z;

    restLenSquared[i] = pDOT(diff, diff);
  }

  // barrier
  double z_ground = -2;
  double contact_area = 1.0 / 2.0;

  stepForwardImplicitEuler(jello, edges, m, restLenSquared, k, timeStep, tolerance, xFlat, z_ground, contact_area);
}
