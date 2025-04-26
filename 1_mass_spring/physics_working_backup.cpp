/*

  USC/Viterbi/Computer Science
  "Jello Cube" Assignment 1 starter code

*/

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

int nodeIndex(int i, int j, int k) {
  return (i * 8 * 8) + (j * 8) + k;
}

// helper functions
void copyJelloToVectors(world *jello, vector<point> &x, vector<point> &vel) {
  x.resize(512);
  vel.resize(512);
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) {
      for(int k = 0; k < 8; k++) {
        int idx = nodeIndex(i,j,k);
        x[idx]   = jello->p[i][j][k];
        vel[idx] = jello->v[i][j][k];
      }
    }
  }
}

void copyVectorsToJello(world *jello, vector<point> &x, vector<point> &vel)
{
  for(int i = 0; i < 8; i++){
    for(int j = 0; j < 8; j++){
      for(int k = 0; k < 8; k++){
        int idx = nodeIndex(i,j,k);
        jello->p[i][j][k].x = x[idx].x;
        jello->p[i][j][k].y = x[idx].y;
        jello->p[i][j][k].z = x[idx].z;
        
        jello->v[i][j][k].x = vel[idx].x;
        jello->v[i][j][k].y = vel[idx].y;
        jello->v[i][j][k].z = vel[idx].z;
      }
    }
  }
}

void copyJelloPositions(world *jello, vector<point> &x) {
  x.resize(512);
  for(int i = 0; i < 8; i++){
    for(int j = 0; j < 8; j++){
      for(int k = 0; k < 8; k++){
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
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) {
      for(int k = 0; k < 8; k++) {
        int idx = nodeIndex(i, j, k);
        // along x
        if(i + 1 < 8) {
          array<int,2> edgeTmp = {idx, nodeIndex(i+1, j, k)};
          edges.push_back(edgeTmp);
        }
        // along y
        if(j + 1 < 8) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j+1, k)};
          edges.push_back(edgeTmp);
        }
        // along z
        if(k + 1 < 8) {
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

  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) {
      for(int k = 0; k < 8; k++) {
        int idx = nodeIndex(i, j, k);
        // Try each offset in the shearOffsets list
        for(auto &ofs : shearOffsets) {
            int ni = i + ofs[0]; // neighbor i
            int nj = j + ofs[1]; // neighbor j
            int nk = k + ofs[2]; // neighbor k

          // check boundaries
          if(ni >= 0 && ni < 8 &&
              nj >= 0 && nj < 8 &&
              nk >= 0 && nk < 8) {
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
  for(int i = 0; i < 8; i++) {
    for(int j = 0; j < 8; j++) {
      for(int k = 0; k < 8; k++) {
        int idx = nodeIndex(i, j, k);
        // i+2
        if(i + 2 < 8) {
          array<int,2> edgeTmp = {idx, nodeIndex(i+2, j, k)};
          edges.push_back(edgeTmp);
        }
        // j+2
        if(j + 2 < 8) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j+2, k)};
          edges.push_back(edgeTmp);
        }
        // k+2
        if(k + 2 < 8) {
          array<int,2> edgeTmp = {idx, nodeIndex(i, j, k+2)};
          edges.push_back(edgeTmp);
        }
      }
    }
  }

  return edges;
}

// Inertia Term
double valInertia(vector<point> &x, vector<point> &xTilde, double m)
{
  double sum = 0.0;
  for (int i = 0; i < 512; i++) {
    point diff = vec_minus_vec(x[i], xTilde[i]);
    sum += 0.5 * m * vec_dot_vec(diff, diff);
  }
  return sum;
}

vector<point> gradInertia(vector<point> &x, vector<point> &xTilde, double m) {
  vector<point> g(512);
  for (int i = 0; i < 512; i++) {
    point diff = vec_minus_vec(x[i], xTilde[i]);
    g[i] = vec_time_scale(diff, m);
  }
  return g;
}

IJV hessInertia(vector<point> &x, vector<point> &xTilde, double m) {
  IJV h;
  h.I.resize(3 * 512);
  h.J.resize(3 * 512);
  h.V.resize(3 * 512);

  for (int i = 0; i < 512; i++) {
    for (int d = 0; d < 3; d++) {
      h.I[i * 3 + d] = i * 3 + d;
      h.J[i * 3 + d] = i * 3 + d;
      h.V[i * 3 + d] = m;
    }
  }
  return h;
}

// Mass-Spring Potential Energy
double valSpring(vector<point> &x, vector< array<int,2> > &e, vector<double> &l2, double k)
{
  double sum = 0.0;
  for(int i = 0; i < e.size(); i++) {
    point diff = vec_minus_vec(x[e[i][0]], x[e[i][1]]);
    sum += l2[i] * 0.5 * k * pow((vec_dot_vec(diff, diff) / l2[i] - 1), 2);
  }
  return sum;
}

vector<point> gradSpring(vector<point> &x, vector< array<int,2> > &e, vector<double> &l2, double k) {
  vector<point> g(512);
    
  // initialize
  for (int i = 0; i < 512; i++){
    pMAKE(0.0, 0.0, 0.0, g[i]);
  }

  for(int i = 0; i < e.size(); i++) {
    point diff = vec_minus_vec(x[e[i][0]], x[e[i][1]]);
    point g_diff = vec_time_scale(diff, 2.0 * k * (vec_dot_vec(diff, diff) / l2[i] - 1));
    g[e[i][0]] = vec_plus_vec(g[e[i][0]], g_diff);
    g[e[i][1]] = vec_minus_vec(g[e[i][1]], g_diff);
  }

  return g;
}

IJV hessSpring(vector<point> &x, vector< array<int,2> > &e, vector<double> &l2, double k) {
  // Each edge contributes 6x6 = 36 entries to the global Hessian
  IJV h;
  h.I.resize(e.size() * 36);
  h.J.resize(e.size() * 36);
  h.V.resize(e.size() * 36);

  for(int i = 0; i < e.size(); i++) {
    // compute diff
    point diff = vec_minus_vec(x[e[i][0]], x[e[i][1]]);
    
    // compute H_diff
    double factor = 2.0 * k / l2[i];
    double Hdiff[9];
    Hdiff[0] = 2.0 * diff.x * diff.x + vec_dot_vec(diff, diff) - l2[i];
    Hdiff[1] = 2.0 * diff.x * diff.y;
    Hdiff[2] = 2.0 * diff.x * diff.z;
    Hdiff[3] = 2.0 * diff.y * diff.x;
    Hdiff[4] = 2.0 * diff.y * diff.y + vec_dot_vec(diff, diff) - l2[i];
    Hdiff[5] = 2.0 * diff.y * diff.z;
    Hdiff[6] = 2.0 * diff.z * diff.x;
    Hdiff[7] = 2.0 * diff.z * diff.y;
    Hdiff[8] = 2.0 * diff.z * diff.z + vec_dot_vec(diff, diff) - l2[i];
    for(int r=0; r<9; r++) {
      Hdiff[r] *= factor;
    }

    // 6x6 block = [[ Hdiff, -Hdiff ],
    //              [ -Hdiff, Hdiff  ]]
    int baseIdx = i * 36; // each edge uses 36 entries
    int n0 = e[i][0];
    int n1 = e[i][1];
    // For local row R in [0..5], local col C in [0..5]:
    //   global row = (R<3 ? n0 : n1)*3 + (R%3)
    //   global col = (C<3 ? n0 : n1)*3 + (C%3)
    //   sign = +1 if (R<3 and C<3) or (R>=3 and C>=3), else -1
    for(int R=0; R<6; R++) {
      for(int C=0; C<6; C++) {
        int nodePartR = (R < 3) ? n0 : n1;
        int nodePartC = (C < 3) ? n0 : n1;

        int coordR = R % 3;
        int coordC = C % 3;

        double sign = ((R<3 && C<3) || (R>=3 && C>=3)) ? 1.0 : -1.0;
        double valRC = sign * Hdiff[coordR*3 + coordC];

        int globalRow = nodePartR * 3 + coordR;
        int globalCol = nodePartC*3 + coordC;

        int offset = baseIdx + (R*6 + C);
        h.I[offset] = globalRow;
        h.J[offset] = globalCol;
        h.V[offset] = valRC;
      }
    }
  }
  return h;
}

// Optimization Time Integrator
double IP_val(vector<point> &x, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h) {
  return valInertia(x, x_tilde, m) + h * h * valSpring(x, e, l2, k);
}

vector<point> IP_grad(vector<point> &x, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h) {
  auto gI = gradInertia(x, x_tilde, m);
  auto gS = gradSpring(x, e, l2, k);

  for(int i=0; i<gI.size(); i++) {
    gI[i].x += h * h * gS[i].x;
    gI[i].y += h * h * gS[i].y;
    gI[i].z += h * h * gS[i].z;
  }
  return gI;
}

IJV IP_hess(vector<point> &x, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h) {
  IJV hIn = hessInertia(x, x_tilde, m);
  IJV hSp = hessSpring(x, e, l2, k);

  for(size_t i=0; i < hSp.V.size(); i++){
      hSp.V[i] *= (h * h);
  }
  
  // combine hIn and hSp
  IJV combined;
  combined.I.reserve(hIn.I.size() + hSp.I.size());
  combined.J.reserve(hIn.J.size() + hSp.J.size());
  combined.V.reserve(hIn.V.size() + hSp.V.size());

  // inertia parts
  combined.I.insert(combined.I.end(), hIn.I.begin(), hIn.I.end());
  combined.J.insert(combined.J.end(), hIn.J.begin(), hIn.J.end());
  combined.V.insert(combined.V.end(), hIn.V.begin(), hIn.V.end());

  // spring parts
  combined.I.insert(combined.I.end(), hSp.I.begin(), hSp.I.end());
  combined.J.insert(combined.J.end(), hSp.J.begin(), hSp.J.end());
  combined.V.insert(combined.V.end(), hSp.V.begin(), hSp.V.end());

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

  // 4) Choose and run a sparse direct solver.
  //    For a typical (semi)definite Hessian in a physics problem,
  //    SimplicialLDLT or SimplicialLLT is common. If not sure, you
  //    can use SparseLU or SparseQR, but they might be slower.
  SimplicialLDLT< SparseMatrix<double> > solver;
  solver.compute(mat);
  if(solver.info() != Success) {
    return vector<double>(n, 0.0);
  }

  VectorXd x = solver.solve(b);
  if(solver.info() != Success) {
    return std::vector<double>(n, 0.0);
  }

  // 5) Convert x back to std::vector<double>
  vector<double> xSol(n);
  for(int i = 0; i < n; i++) {
    xSol[i] = x[i];
  }
  return xSol;
}

vector<point> search_dir(vector<point> &x, vector<point> &x_tilde, double m, vector< array<int,2> > &e, vector<double> &l2, double k, double h) {
  IJV A = IP_hess(x, x_tilde, m, e, l2, k, h);
  vector<point> g = IP_grad(x, x_tilde, m, e, l2, k, h);

  // Flatten g into -g for the RHS
  int N = g.size();
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

void stepForwardImplicitEuler(world *jello, vector< array<int,2> > &edges, double m, vector<double> &l2, double k, double h, double tol) {
  // x, v, x_old
  vector<point> x, v; copyJelloToVectors(jello, x, v);
  vector<point> x_old = x;

  // x_tilde
  vector<point> x_tilde(x.size());
  for(size_t i=0; i < x.size(); i++) {
    x_tilde[i].x = x[i].x + h * v[i].x;
    x_tilde[i].y = x[i].y + h * v[i].y;
    x_tilde[i].z = x[i].z + h * v[i].z;
  }

  // newton loop
  int iter = 0;
  double E_last = IP_val(x, x_tilde, m, edges, l2, k, h);
  vector<point> p = search_dir(x, x_tilde, m, edges, l2, k, h);

  while(normInfOverH(p, h) > tol && iter < 30) {
    
    // line search
    double alpha = 1.0;
    while(true) {
      vector<point> tempX(x.size());
      for(size_t i=0; i < x.size(); i++){
        tempX[i].x = x[i].x + alpha * p[i].x;
        tempX[i].y = x[i].y + alpha * p[i].y;
        tempX[i].z = x[i].z + alpha * p[i].z;
      }
      double E_try = IP_val(tempX, x_tilde, m, edges, l2, k, h);
      if(E_try <= E_last){
        // accept
        break;
      }
      alpha *= 0.5;
      if(alpha < 1e-12) {
        break;
      }
    }

    // update x
    for(size_t i=0; i < x.size(); i++){
        x[i].x += alpha * p[i].x;
        x[i].y += alpha * p[i].y;
        x[i].z += alpha * p[i].z;
    }
    E_last = IP_val(x, x_tilde, m, edges, l2, k, h);
    p = search_dir(x, x_tilde, m, edges, l2, k, h);
    iter++;
  }

  // v = (x - x_n) / h
  for(int i=0; i < x.size(); i++) {
    v[i].x = (x[i].x - x_old[i].x) / h;
    v[i].y = (x[i].y - x_old[i].y) / h;
    v[i].z = (x[i].z - x_old[i].z) / h;
  }

  // write results back to jello->p and jello->v
  copyVectorsToJello(jello, x, v);
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

  stepForwardImplicitEuler(jello, edges, m, restLenSquared, k, timeStep, tolerance);
}
