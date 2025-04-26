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

static const int GRID_SIZE = 8;

// inline function to convert 3D (i, j, k) to a single index [0..511].
inline int nodeIndex(int i, int j, int k)
{
  return (i * 8 * 8) + (j * 8) + k;
}

/**
 * Generate edges for the jello cube (8x8x8).
 * Returns a list of edges, where each edge is an array of two node indices.
 * 
 * Each node is indexed by nodeIndex(i, j, k).
 **/
std::vector< std::array<int, 2> > generateJelloEdges()
{
    std::vector< std::array<int, 2> > edges;
    edges.reserve(6000); 
    // Reserve some space to avoid repeated allocations. 6000 is a guess that
    // comfortably accommodates all possible edges.

    // --- 1) Structural edges -----------------------------------------------
    //    (i, j, k) -> (i+1, j, k), (i, j+1, k), (i, j, k+1)
    for(int i = 0; i < GRID_SIZE; i++){
        for(int j = 0; j < GRID_SIZE; j++){
            for(int k = 0; k < GRID_SIZE; k++){
                int idx = nodeIndex(i, j, k);

                // along x
                if(i + 1 < GRID_SIZE)
                {
                    edges.push_back({ idx, nodeIndex(i+1, j, k) });
                }
                // along y
                if(j + 1 < GRID_SIZE)
                {
                    edges.push_back({ idx, nodeIndex(i, j+1, k) });
                }
                // along z
                if(k + 1 < GRID_SIZE)
                {
                    edges.push_back({ idx, nodeIndex(i, j, k+1) });
                }
            }
        }
    }

    // --- 2) Shear edges ----------------------------------------------------
    // We'll connect diagonally within the xy, yz, xz planes plus
    // the main diagonals inside each small 3D cube.
    // For each (i,j,k), we look at potential offsets that represent
    // "one step in one axis, and one step in another axis" => face diagonals
    // and "one step in each axis" => 3D cell diagonal.
    //
    // The offsets we can consider (dx, dy, dz) in { -1, 1 }, but not all 3 zero,
    // typically focusing on "positive" direction or ensuring i < i+dx to avoid duplicates.

    // Offsets for shear inside faces or cubes:
    // For example, consider these offsets:
    //  (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)
    //  And the 3D diagonals: (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1) ...
    // 
    // One simpler approach: systematically check all combinations of dx, dy, dz in -1..1
    // that have exactly two or three nonzero coordinates. But for demonstration, let's do
    // them in a direct enumerative way and check boundaries.

    // We'll systematically define an array of possible shear offsets.
    // (In practice, you might break these into plane-specific or diagonal logic.)
    std::vector< std::array<int,3> > shearOffsets = {
        // Face diagonals, XY plane (dz=0)
        {1, 1, 0},
        {1,-1, 0},
        {-1, 1, 0},
        {-1,-1, 0},
        // Face diagonals, YZ plane (dx=0)
        {0, 1, 1},
        {0, 1,-1},
        {0,-1, 1},
        {0,-1,-1},
        // Face diagonals, XZ plane (dy=0)
        {1, 0, 1},
        {1, 0,-1},
        {-1,0, 1},
        {-1,0,-1},
        // 3D diagonals
        {1, 1, 1},
        {1, 1,-1},
        {1,-1, 1},
        {-1,1, 1},
        { -1,-1,1},
        { -1,1,-1},
        { 1,-1,-1},
        { -1,-1,-1}
    };

    for(int i = 0; i < GRID_SIZE; i++)
    {
        for(int j = 0; j < GRID_SIZE; j++)
        {
            for(int k = 0; k < GRID_SIZE; k++)
            {
                int idx = nodeIndex(i, j, k);

                // Try each offset in the shearOffsets list
                for(auto &ofs : shearOffsets)
                {
                    int ni = i + ofs[0]; // neighbor i
                    int nj = j + ofs[1]; // neighbor j
                    int nk = k + ofs[2]; // neighbor k

                    // check boundaries
                    if(ni >= 0 && ni < GRID_SIZE &&
                       nj >= 0 && nj < GRID_SIZE &&
                       nk >= 0 && nk < GRID_SIZE)
                    {
                        // To avoid duplicating edges in both directions,
                        // only push if (ni, nj, nk) > (i, j, k) in flattened order.
                        // That is, we only add edge if new_idx > idx.
                        int new_idx = nodeIndex(ni, nj, nk);
                        if(new_idx > idx)
                        {
                            edges.push_back({ idx, new_idx });
                        }
                    }
                }
            }
        }
    }

    // --- 3) Bend edges -----------------------------------------------------
    // (i, j, k) -> (i+2, j, k), (i, j+2, k), (i, j, k+2)
    // as long as they're in range.
    for(int i = 0; i < GRID_SIZE; i++)
    {
        for(int j = 0; j < GRID_SIZE; j++)
        {
            for(int k = 0; k < GRID_SIZE; k++)
            {
                int idx = nodeIndex(i, j, k);

                // i+2
                if(i + 2 < GRID_SIZE)
                {
                    edges.push_back({ idx, nodeIndex(i+2, j, k) });
                }
                // j+2
                if(j + 2 < GRID_SIZE)
                {
                    edges.push_back({ idx, nodeIndex(i, j+2, k) });
                }
                // k+2
                if(k + 2 < GRID_SIZE)
                {
                    edges.push_back({ idx, nodeIndex(i, j, k+2) });
                }
            }
        }
    }

    return edges;
}

//------------------------------------------------------------------------------
// grad(): Computes the gradient, which is a vector of size x.size(),
// each entry is: m[i] * (x[i] - xTilde[i]).
//------------------------------------------------------------------------------
std::vector<Vec3> grad(const std::vector<Vec3> &x,
                       const std::vector<Vec3> &xTilde,
                       const std::vector<double> &m)
{
    // Return a vector of the same size as x, each element is a Vec3.
    std::vector<Vec3> g(x.size());
    for (size_t i = 0; i < x.size(); i++)
    {
        Vec3 diff = x[i] - xTilde[i];
        g[i] = diff * m[i];
    }
    return g;
}

//------------------------------------------------------------------------------
// hess(): Returns the Hessian in IJV format.
// For a purely diagonal block, the Hessian for each node i in 3D is
//    m[i] * I_{3x3},
// i.e. the diagonal entries (3i+d, 3i+d) for d = 0..2 each have value m[i].
//
// If we flatten (x[0], x[1], ..., x[n-1]) into a 3n-dimensional vector
// [ x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, ... ], each node i spans
// indices 3i, 3i+1, and 3i+2 in that flattened vector.
//------------------------------------------------------------------------------
IJV hess(const std::vector<Vec3> &x,
         const std::vector<Vec3> &xTilde,
         const std::vector<double> &m)
{
    size_t n = x.size();

    // We will have exactly 3*n non-zero entries on the diagonal (each coordinate).
    IJV hessian;
    hessian.I.resize(3*n);
    hessian.J.resize(3*n);
    hessian.V.resize(3*n);

    for (size_t i = 0; i < n; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            // Flattened index:
            int rowColIdx = static_cast<int>(i * 3 + d);

            hessian.I[rowColIdx] = rowColIdx; // row
            hessian.J[rowColIdx] = rowColIdx; // column
            hessian.V[rowColIdx] = m[i];      // diagonal value
        }
    }

    return hessian;
}

// find neighbour points for structural
void findNeighbours(struct world * jello, int i, int j, int k, struct point neighbours[32], struct point Vneighbours[32]){
  int ip, jp, kp;
  int structural_neighbours[32][3] = { // total 6+20+6=32 neighbours
    // structual
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {-1, 0, 0}, {0, -1, 0}, {0, 0, -1},
    // shear
    {1, 1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, -1, 0},
    {0, 1, 1}, {0, -1, 1}, {0, -1, -1}, {0, 1, -1},
    {1, 0, 1}, {-1, 0, 1}, {-1, 0, -1}, {1, 0, -1},
    {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, -1}, {1, -1, -1},
    // bend
    {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
    {-2, 0, 0}, {0, -2, 0}, {0, 0, -2}
  };
  for (int n=0; n<32; n++){
    ip = i + structural_neighbours[n][0];
    jp = j + structural_neighbours[n][1];
    kp = k + structural_neighbours[n][2];
    if (!(ip>7 || ip<0 || jp>7 || jp<0 || kp>7 || kp<0)){
      neighbours[n] = jello->p[ip][jp][kp];
      Vneighbours[n] = jello->v[ip][jp][kp];
    } else { // when neighbour does not exist
      neighbours[n] = jello->p[i][j][k];
      Vneighbours[n] = jello->v[i][j][k];
    }
  }
}

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

//------------------------------------------------------------------------------
// val(): Computes the scalar energy:
//    0.5 * sum_i [ m[i] * || x[i] - xTilde[i] ||^2 ]
//------------------------------------------------------------------------------
double val(const std::vector<Vec3> &x,
           const std::vector<Vec3> &xTilde,
           const std::vector<double> &m)
{
    // Assumes x.size() == xTilde.size() == m.size().
    double sumVal = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        Vec3 diff = x[i] - xTilde[i];
        double sqDist = diff.dot(diff); // || diff ||^2
        sumVal += 0.5 * m[i] * sqDist;
    }
    return sumVal;
}

// compute inertia for one point
double inertia_val(struct world * jello, struct point x_tilde[8][8][8]) {
  double m = jello->mass;

  double sum = 0.0; 
  for (int i=0; i<=7; i++){
    for (int j=0; j<=7; j++){
      for (int k=0; k<=7; k++){
        struct point diff;
        pDIFFERENCE(jello->p[i][j][k], x_tilde[i][j][k], diff);
        sum += 0.5 * m * (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
      }
    }
  }
  return sum;
}

void inertia_grad(struct world * jello, struct point x_tilde[8][8][8], struct point g[8][8][8]) {
  double m = jello->mass;

  for (int i=0; i<=7; i++){
    for (int j=0; j<=7; j++){
      for (int k=0; k<=7; k++){
        g[i][j][k] = vec_time(vec_minus(jello->p[i][j][k], x_tilde[i][j][k]), m);
      }
    }
  }
}

void inertia_hess(struct world * jello, double H[3][3]) {
  double m = jello->mass;

  H[0][0] = m;   H[0][1] = 0.0; H[0][2] = 0.0;
  H[1][0] = 0.0; H[1][1] = m;   H[1][2] = 0.0;
  H[2][0] = 0.0; H[2][1] = 0.0; H[2][2] = m;
}

// compute mass-spring potential energy for one point
double mass_spring_val(struct world * jello, int i, int j, int k, struct point x) {
  double kE = jello->kElastic;
  struct point neighbours[32];
  struct point Vneighbours[32];
  findNeighbours(jello, i, j, k, neighbours, Vneighbours);

  double sum = 0.0;
  for (int n=0; n<32; n++){
    // find spring rest length
    double l;
    if (n < 6) {
      l = 1.0 / 7.0;
    } else if (n < 18){
      l = sqrt(2.0) / 7.0;
    } else if (n < 26){
      l = sqrt(3.0) / 7.0;
    } else {
      l = 2.0 / 7.0;
    }
    
    struct point e = neighbours[n];
    struct point diff; pDIFFERENCE(x, e, diff);

    sum += l * l * 0.5 * kE * pow(((diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) / (l * l)) - 1, 2);
  }
  return sum;
}

struct point mass_spring_grad(struct world * jello, int i, int j, int k, struct point x){
  double kE = jello->kElastic;
  struct point neighbours[32];
  struct point Vneighbours[32];
  findNeighbours(jello, i, j, k, neighbours, Vneighbours);

  struct point g; pMAKE(0.0, 0.0, 0.0, g);
  for (int n=0; n<32; n++){
    // find spring rest length
    double l;
    if (n < 6) {
      l = 1.0 / 7.0;
    } else if (n < 18){
      l = sqrt(2.0) / 7.0;
    } else if (n < 26){
      l = sqrt(3.0) / 7.0;
    } else {
      l = 2.0 / 7.0;
    }
    
    struct point e = neighbours[n];
    struct point diff; pDIFFERENCE(x, e, diff);
    pMULTIPLY(diff, 2 * kE * (((diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) / (l * l)) - 1), diff);
    pSUM(g, diff, g);
  }
  return g;
}

void mass_spring_hess(struct world * jello, int i, int j, int k, struct point x, double H[3][3]){
  double kE = jello->kElastic;
  struct point neighbours[32];
  struct point Vneighbours[32];
  findNeighbours(jello, i, j, k, neighbours, Vneighbours);

  // set every value of H to 0.0
  for(int r = 0; r < 3; r++){
    for(int c = 0; c < 3; c++){
      H[r][c] = 0.0;
    }
  }

  for (int n = 0; n < 32; n++){
    // find spring rest length
    double l;
    if (n < 6) {
      l = 1.0 / 7.0;
    } else if (n < 18){
      l = sqrt(2.0) / 7.0;
    } else if (n < 26){
      l = sqrt(3.0) / 7.0;
    } else {
      l = 2.0 / 7.0;
    }
    
    struct point e = neighbours[n];
    struct point diff; pDIFFERENCE(x, e, diff);
    double diff_dot = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    double constant1 = ((2 * kE) / (l * l)) * (diff_dot - (l * l));
    double constant2 = ((4 * kE) / (l * l));
    // row 1
    H[0][0] += constant1 + (constant2 * diff.x * diff.x);
    H[0][1] += constant2 * diff.x * diff.y;
    H[0][2] += constant2 * diff.x * diff.z;
    // row 2
    H[1][0] += constant2 * diff.y * diff.x;
    H[1][1] += constant1 + (constant2 * diff.y * diff.y);
    H[1][2] += constant2 * diff.y * diff.z;
    // row 3
    H[2][0] += constant2 * diff.z * diff.x;
    H[2][1] += constant2 * diff.z * diff.y;
    H[2][2] += constant1 + (constant2 * diff.z * diff.z);
  }
}

double IP_val(struct world * jello, int i, int j, int k, struct point x, struct point x_tilde) {
  double h = jello->dt;
  return inertia_val(jello, x, x_tilde) + h * h * mass_spring_val(jello, i, j, k, x);
}

struct point IP_grad(struct world * jello, int i, int j, int k, struct point x, struct point x_tilde) {
  double h = jello->dt;
  struct point result, g_inertia, g_mass_spring;

  g_inertia = inertia_grad(jello,  x, x_tilde);
  g_mass_spring = mass_spring_grad(jello, i, j, k, x);

  pMULTIPLY(g_mass_spring, h * h, g_mass_spring)
  pSUM(g_inertia, g_mass_spring, result);
  return result;
}

void IP_hess(struct world * jello, int i, int j, int k, struct point x, double H[3][3]) {
  double H_inertia[3][3], H_spring[3][3];
  inertia_hess(jello, H_inertia);
  for(int r=0; r<3; r++){
    for(int c=0; c<3; c++){
      // std::cout << " H_inertia[r][c]: " << r << " , " << c << " = " << H_inertia[r][c];
    }
  }
  mass_spring_hess(jello, i, j, k, x, H_spring);

  double h = jello->dt;
  for(int r=0; r<3; r++){
    for(int c=0; c<3; c++){
      H[r][c] = H_inertia[r][c] + (h * h * H_spring[r][c]);
      // std::cout << " H[r][c]: " << r << " , " << c << " = " << H[r][c];
    }
  }
}

void find_inverse_matrix(double H[3][3], double H_inverse[3][3]) {
  double det = H[0][0] * H[1][1] * H[2][2] 
             + H[0][1] * H[1][2] * H[2][0] 
             + H[0][2] * H[1][0] * H[2][1]
             - H[0][0] * H[1][2] * H[2][1]
             - H[0][1] * H[1][0] * H[2][2]
             - H[0][2] * H[1][1] * H[2][0];

  // std::cout << " det: " << det;

  double adj[3][3];
  // row 1
  adj[0][0] = H[1][1] * H[2][2] - H[2][1] * H[1][2];
  adj[0][1] = -(H[0][1] * H[2][2] - H[2][1] * H[0][2]);
  adj[0][2] = H[0][0] * H[1][1] - H[1][0] * H[0][1];
  // row 2
  adj[1][0] = -(H[1][0] * H[2][2] - H[2][0] * H[1][2]);
  adj[1][1] = H[0][0] * H[2][2] - H[2][0] * H[0][2];
  adj[1][2] = -(H[0][0] * H[1][2] - H[1][0] * H[0][2]);
  // row 3
  adj[2][0] = H[1][0] * H[2][1] - H[2][0] * H[1][1];
  adj[2][1] = -(H[0][0] * H[2][1] - H[2][0] * H[0][1]);
  adj[2][2] = H[0][0] * H[1][1] - H[1][0] * H[0][1];

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      H_inverse[i][j] = adj[i][j] / det;
    }
  }
}

void search_dir(struct world * jello, int i, int j, int k, struct point x, struct point x_tilde, double X[3]) {
  double H[3][3];
  IP_hess(jello, i, j, k, x, H);
  // std::cout << " H[][] " << H[0][0];
  double H_inverse[3][3];
  find_inverse_matrix(H, H_inverse);


  struct point G = IP_grad(jello, i, j, k, x, x_tilde);

  X[0] = H_inverse[0][0] * G.x + H_inverse[0][1] * G.y + H_inverse[0][2] * G.z;
  X[1] = H_inverse[1][0] * G.x + H_inverse[1][1] * G.y + H_inverse[1][2] * G.z;
  X[2] = H_inverse[2][0] * G.x + H_inverse[2][1] * G.y + H_inverse[2][2] * G.z;
}

double find_inf_norm(double p[3]) {
  double maximum = 0;
  for (int i = 0; i < 3; i++) {
    if (abs(p[i]) > maximum) {
      maximum = abs(p[i]);
    }
  }
  return maximum;
}

// compute Dampling Force for point A
struct point computeDampling(struct world * jello, int i, int j, int k){
  struct point a = jello->p[i][j][k]; // A
  struct point b; // B
  struct point F; F.x=0; F.y=0; F.z=0;// Dampling force
  struct point L; // L = A - B
  struct point Lnorm; // L / |L|
  double length; // |L|
  double kDampling = jello->dElastic;

  // find neighbour
  struct point neighbours[32];
  struct point Vneighbours[32];
  findNeighbours(jello, i, j, k, neighbours, Vneighbours);

  // velocity
  struct point Va = jello->v[i][j][k];
  struct point Vb;
  struct point Vdiff;

  for (int n=0; n<32; n++){
    b = neighbours[n];
    Vb = Vneighbours[n];
    pDIFFERENCE(Va, Vb, Vdiff); // compute Va - Vb
    pDIFFERENCE(a, b, L); // compute L = a - b
    
    // skip if neighbour does not exist
    if (L.x == 0 && L.y == 0 && L.z ==0){
      continue;
    }

    pCPY(L, Lnorm) 
    pNORMALIZE(Lnorm); // compute |L|
    double dotproduct = -kDampling * ((Vdiff.x * L.x + Vdiff.y * L.y + Vdiff.z * L.z) / length);
    struct point F1;
    pMULTIPLY(Lnorm, dotproduct, F1);
    pSUM(F1, F, F);
  }
  return F;
}

struct point computeCollisionDetection(struct world * jello, int i, int j, int k){
  struct point a = jello->p[i][j][k];
  struct point v = jello->v[i][j][k];
  double kCollision = jello->kCollision;
  double dCollision = jello->dCollision;
  double penetration;
  struct point normal;
  struct point Fhook, Fdamping, Ftotal;
  double length;

  // compute Hook Force
  if (a.x <= -2){
    pMAKE(1, 0, 0, normal);
    penetration = -2 - a.x;
  } else if (a.x >= 2) {
    pMAKE(-1, 0, 0, normal);
    penetration = a.x - 2;
  } else if (a.y <= -2) {
    pMAKE(0, 1, 0, normal);
    penetration = -2 - a.y;
  } else if (a.y >= 2) {
    pMAKE(0, -1, 0, normal);
    penetration = a.y - 2;
  } else if (a.z <= -2) {
    pMAKE(0, 0, 1, normal);
    penetration = -2 - a.z;
  } else if (a.z >= 2) {
    pMAKE(0, 0, -1, normal);
    penetration = a.z - 2;
  }
  pMULTIPLY(normal, (kCollision * penetration), Fhook);

  // compute Damping Force
  pMULTIPLY(normal, (dCollision * (v.x * normal.x + v.y * normal.y + v.z * normal.z)), Fdamping);

  // compute total collision detection force
  pSUM(Fhook, Fdamping, Ftotal);
  return Ftotal;
}

// detect collision detection and compute force if collide
bool detectCollisionDetection(struct world * jello, int i, int j, int k){
  struct point a = jello->p[i][j][k];
  if (a.x <= -2 || a.x >= 2 
  || a.y <= -2 || a.y >= 2
  || a.z <= -2 || a.z >= 2){
    return true;
  }
  return false;
}

// I used Trilinear Interpolation, equations are referenced from https://spie.org/samples/PM159.pdf
struct point computeForceField(struct world * jello, int i, int j, int k){
  struct point a = jello->p[i][j][k];
  int n = jello->resolution;
  struct point * forceField = jello->forceField;

  double x, y, z; // actual space location
  x = a.x;
  y = a.y;
  z = a.z;
  if (x < -2.0) { x = -2.0; } else if (x > 2.0) { x = 2.0; } // avoid out of bounding box
  if (y < -2.0) { y = -2.0; } else if (y > 2.0) { y = 2.0; }
  if (z < -2.0) { z = -2.0; } else if (z > 2.0) { z = 2.0; }

  double u, v, w; // index
  u = ((x + 2.0) * (n - 1.0)) / 4.0;
  v = ((y + 2.0) * (n - 1.0)) / 4.0;
  w = ((z + 2.0) * (n - 1.0)) / 4.0;
  int u0 = floor(u);
  int u1 = ceil(u);
  int v0 = floor(v);
  int v1 = ceil(v);
  int w0 = floor(w);
  int w1 = ceil(w);
  
  struct point p000 = jello->forceField[u0*n*n + v0*n + w0];
  struct point p100 = jello->forceField[u1*n*n + v0*n + w0];
  struct point p110 = jello->forceField[u1*n*n + v1*n + w0];
  struct point p010 = jello->forceField[u0*n*n + v1*n + w0];
  struct point p001 = jello->forceField[u0*n*n + v0*n + w1];
  struct point p011 = jello->forceField[u0*n*n + v1*n + w1];
  struct point p111 = jello->forceField[u1*n*n + v1*n + w1];
  struct point p101 = jello->forceField[u1*n*n + v0*n + w1];

  struct point c0 = p000;
  struct point c1; pDIFFERENCE(p100, p000, c1);
  struct point c2; pDIFFERENCE(p010, p000, c2);
  struct point c3; pDIFFERENCE(p001, p000, c3);
  struct point c4;
  c4.x = p110.x - p010.x - p100.x + p000.x;
  c4.y = p110.y - p010.y - p100.y + p000.y;
  c4.z = p110.z - p010.z - p100.z + p000.z;
  struct point c5;
  c5.x = p011.x - p001.x - p010.x + p000.x;
  c5.y = p011.y - p001.y - p010.y + p000.y;
  c5.z = p011.z - p001.z - p010.z + p000.z;
  struct point c6;
  c6.x = p101.x - p001.x - p100.x + p000.x;
  c6.y = p101.y - p001.y - p100.y + p000.y;
  c6.z = p101.z - p001.z - p100.z + p000.z;
  struct point c7;
  c7.x = p111.x - p011.x - p101.x - p110.x + p100.x + p001.x + p010.x - p000.x;
  c7.y = p111.y - p011.y - p101.y - p110.y + p100.y + p001.y + p010.y - p000.y;
  c7.z = p111.z - p011.z - p101.z - p110.z + p100.z + p001.z + p010.z - p000.z;

  double u_delta, v_delta, w_delta; 
  if (u1 == u0) {
    u_delta = 0.0;
  } else {
    u_delta = (u - u0) / (u1 - u0);
  }
  if (v1 == v0) {
    v_delta = 0.0;
  } else {
    v_delta = (v - v0) / (v1 - v0);
  }
  if (w1 == w0) {
    w_delta = 0.0;
  } else {
    w_delta = (w - w0) / (w1 - w0);
  }
  
  struct point F;
  F.x = c0.x + c1.x * u_delta + c2.x * v_delta + c3.x * w_delta + c4.x * u_delta * v_delta + c5.x * v_delta * w_delta + c6.x * w_delta * u_delta + c7.x * u_delta * v_delta * w_delta;
  F.y = c0.y + c1.y * u_delta + c2.y * v_delta + c3.y * w_delta + c4.y * u_delta * v_delta + c5.y * v_delta * w_delta + c6.y * w_delta * u_delta + c7.y * u_delta * v_delta * w_delta;
  F.z = c0.z + c1.z * u_delta + c2.z * v_delta + c3.z * w_delta + c4.z * u_delta * v_delta + c5.z * v_delta * w_delta + c6.z * w_delta * u_delta + c7.z * u_delta * v_delta * w_delta;

  return F;
}

/* performs one step of Euler Integration */
/* as a result, updates the jello structure */
void IPC(struct world * jello)
{ 
  std::cout << "using ruler";
  int i,j,k;
  // notice we are missing Fforcefield right now !!!
  double m = jello->mass;
  double h = jello->dt;
  struct point acc;
  struct point Ftotal, Fhook, Fdampling, Ffield, Fcollision;
  
  for (i=0; i<=7; i++){
    for (j=0; j<=7; j++){
      for (k=0; k<=7; k++){
        struct point x = jello->p[i][j][k];
        struct point x_old; pCPY(x, x_old);
        struct point x_new; pCPY(x, x_new);
        struct point v = jello->v[i][j][k];
        struct point x_tilde; pMULTIPLY(v, h, x_tilde); pSUM(x, x_tilde, x_tilde);
        

        // Newton loop
        int iter = 0;
        double E_last = IP_val(jello, i, j, k, x, x_tilde);
        double p[3]; search_dir(jello, i, j, k, x, x_tilde, p);
        double p_inf_norm = find_inf_norm(p);
        while (p_inf_norm / h > 1e-2 && iter <= 30) {
          // line search
          double alpha = 1;
          
          struct point x_alpha;
          x_alpha.x = x_new.x + alpha * p[0];
          x_alpha.y = x_new.y + alpha * p[1];
          x_alpha.z = x_new.z + alpha * p[2];

          while (IP_val(jello, i, j, k, x_alpha, x_tilde) > E_last) {
            alpha = alpha / 2;
            x_alpha.x = x_new.x + alpha * p[0];
            x_alpha.y = x_new.y + alpha * p[1];
            x_alpha.z = x_new.z + alpha * p[2];
          }

          x_new.x += alpha * p[0];
          x_new.y += alpha * p[1];
          x_new.z += alpha * p[2];

          E_last = IP_val(jello, i, j, k, x_new, x_tilde);
          search_dir(jello, i, j, k, x_new, x_tilde, p);
          
          p_inf_norm = find_inf_norm(p);
          iter += 1;
        }
        // std::cout << "new point";
        jello->p[i][j][k].x = x_new.x;
        jello->p[i][j][k].y = x_new.y;
        jello->p[i][j][k].z = x_new.z;

        struct point v_new;
        pDIFFERENCE(x_new, x_old, v_new);
        jello->v[i][j][k].x = v_new.x / h;
        jello->v[i][j][k].y = v_new.y / h;
        jello->v[i][j][k].z = v_new.z / h;
      }
    }
  }
}