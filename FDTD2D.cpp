#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace std;

#define grid_row 200
#define grid_col 200

void initialize_parameters(double *gaz, double *gi2, double *gi3, double *fi2,
                           double *fi3, double *gj2, double *gj3, double *fj2,
                           double *fj3);
void creat_PML(double *gi2, double *gi3, double *gj2, double *gj3,
               double *fi1, double *fi2, double *fi3, double *fj1,
               double *fj2, double *fj3, int npml);
void creat_cylinders(pair<int, int> *centers, int num, double *gaz, double *gbz,
                     double radius, double epsr, double sigma, double delta_t,
                     double epsz);
void calculateDz(double *dz, double *hx, double *hy, double *gi3, double *gj3,
                 double *gi2, double *gj2);
void calculateEz(double *ez, double *dz, double *gaz, double *gbz, double *iz);
void calculateHx(double *ez, double *ihx, double *hx, double *fi1, double *fj2, double *fj3);
void calculateHy(double *ez, double *hy, double *ihy, double *fi2, double *fi3, double *fj1);
int main(){

    const int ia = 10;
    const int ib = grid_row - ia - 1;
    const int ja = 10;
    const int jb = grid_col - ja - 1;

    const double delta_x = 0.0001;
    const double delta_t = delta_x / 6e8;

    const double epsz = 8.854e-12;

    double gaz[grid_row][grid_col] = {};
    double gbz[grid_row][grid_col] = {};

    double ez[grid_row][grid_col] = {};
    double dz[grid_row][grid_col] = {};
    double hx[grid_row][grid_col] = {};
    double hy[grid_row][grid_col] = {};

    double iz[grid_row][grid_col] = {};
    double ihx[grid_row][grid_col] = {};
    double ihy[grid_row][grid_col] = {};
    double ez_inc[grid_col] = {};
    double hx_inc[grid_col] = {};

    // PML parameters
    const int npml = 10;
    double boundary_low[] = {0, 0};
    double boundary_high[] = {0, 0};
    double gi2[grid_row] = {};
    double gi3[grid_row] = {};
    double fi1[grid_row] = {};
    double fi2[grid_row] = {};
    double fi3[grid_row] = {};
    double gj2[grid_col] = {};
    double gj3[grid_col] = {};
    double fj1[grid_col] = {};
    double fj2[grid_col] = {};
    double fj3[grid_col] = {};

    // Dielectric area parameter
    const double epsr = 30;
    const double sigma = 0.3;
    const double radius = 5;
    pair<int,int>centers[] = {
        {50, 50}, {50, 100}, {50, 150},
        {100, 50}, {100, 100}, {100, 150},
        {150, 50}, {150, 100}, {150, 150}
    };

    // pulse parameters
    const int t0 = 20;
    const int spread = 8;
    double pulse = 0;

    initialize_parameters((double *)gaz, gi2, gi3, fi2,
                           fi3, gj2, gj3, fj2, fj3);
    creat_PML(gi2, gi3, gj2, gj3, fi1, fi2, fi3, fj1,fj2, fj3, npml);
    creat_cylinders(centers, sizeof(centers)/sizeof(centers[1]), (double *)gaz,
                    (double *)gbz, radius, epsr, sigma, delta_t, epsz);

    const int nsteps = 2000;

    // Main FDTD loop
    for(int time_step=1; time_step<=nsteps; time_step++){

        auto start = std::chrono::high_resolution_clock::now();

        // Incident Ez values
        for(int j=1; j<grid_col; j++){
            ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j-1] - hx_inc[j]);
        }

        // Absorbing Boundary Conditions
        ez_inc[0] = boundary_low[0];
        boundary_low[0]=boundary_low[1];
        boundary_low[1]=ez_inc[1];

        ez_inc[grid_col-1] = boundary_high[0];
        boundary_high[0]=boundary_high[1];
        boundary_high[1]=ez_inc[grid_col-2];


        calculateDz((double *)dz, (double *)hx, (double *)hy, gi3, gj3,
                gi2, gj2);

        // Source
        pulse = exp(-0.5 * pow(((t0-time_step) / spread), 2));
        ez_inc[3] = pulse;

        // Incident Dz values
        for(int i=ia; i<=ib; i++){
            dz[i][ja] = dz[i][ja] + 0.5 * hx_inc[ja-1];
            dz[i][jb] = dz[i][jb] - 0.5 * hx_inc[jb];
        }

        calculateEz((double *)ez, (double *)dz, (double *)gaz, (double *)gbz, (double *)iz);

        // Calculate the Incident Hx
        for(int j=0; j<grid_col-1; j++){
            hx_inc[j] = hx_inc[j] + 0.5 * (ez_inc[j] - ez_inc[j+1]);
        }

        calculateHx((double *)ez, (double *)ihx, (double *)hx, fi1, fj2, fj3);

        // Incident Hx values
        for (int i=ia; i<=ib; i++) {
            hx[i][ja-1] = hx[i][ja-1] + 0.5 * ez_inc[ja];
            hx[i][jb] = hx[i][jb] - 0.5 * ez_inc[jb];
        }

        calculateHy((double *)ez, (double *)hy, (double *)ihy, fi2, fi3, fj1);

        // Incident Hy value
        for (int j=ja; j<=jb; j++) {
            hy[ia-1][j] = hy[ia-1][j] - 0.5 * ez_inc[j];
            hy[ib][j] = hy[ib][j] + 0.5 * ez_inc[j];
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        ofstream outfile;
        outfile.open("execution_time_CPU.txt", ios::app);
        outfile << duration.count() << " ";
        outfile.close();

        outfile.open("data_cpp.txt", ios::app);
        for(int i=0; i<grid_col; i++){
            for(int j=0; j<grid_row; j++){
                if(j!=grid_row-1){
                    outfile << setprecision(4) << fixed << ez[i][j] << " ";
                } else{
                    outfile << setprecision(4) << fixed << ez[i][j] << endl;
                }
            }
        }
        outfile.close();
    }

    return 0;
}
void initialize_parameters(double *gaz, double *gi2, double *gi3, double *fi2,
                           double *fi3, double *gj2, double *gj3, double *fj2,
                           double *fj3){
    for(int i=0; i<grid_row; i++){
        for(int j=0; j<grid_col; j++){
            gaz[i*grid_col+j] = 1;
        }
    }
    for(int n=0; n<grid_row; n++){
        gi2[n] = 1;
        gi3[n] = 1;
        fi2[n] = 1;
        fi3[n] = 1;
    }
    for(int n=0; n<grid_col; n++){
        gj2[n] = 1;
        gj3[n] = 1;
        fj2[n] = 1;
        fj3[n] = 1;
    }
}

void creat_PML(double *gi2, double *gi3, double *gj2, double *gj3,
               double *fi1, double *fi2, double *fi3, double *fj1,
               double *fj2, double *fj3, int npml){
    for(int n=0; n<npml; n++){
        double xnum = npml - n;
        double xxn = xnum / npml;
        double xn = 0.33 * pow(xxn, 3);
        gi2[n] = 1 / (1 + xn);
        gi2[grid_row-1-n] = 1 / (1 + xn);
        gi3[n] = (1 - xn) / (1 + xn);
        gi3[grid_row-1-n] = (1 - xn) / (1 + xn);
        gj2[n] = 1 / (1 + xn);
        gj2[grid_col-1-n] = 1 / (1 + xn);
        gj3[n] = (1 - xn) / (1 + xn);
        gj3[grid_col-1-n] = (1 - xn) / (1 + xn);
        xxn = (xnum - 0.5) / npml;
        xn = 0.33 * pow(xxn, 3);
        fi1[n] = xn;
        fi1[grid_row-2-n] = xn;
        fi2[n] = 1 / (1 + xn);
        fi2[grid_row-2-n] = 1 / (1 + xn);
        fi3[n] = (1 - xn) / (1 + xn);
        fi3[grid_row-2-n] = (1 - xn) / (1 + xn);
        fj1[n] = xn;
        fj1[grid_col-2-n] = xn;
        fj2[n] = 1 / (1 + xn);
        fj2[grid_col-2-n] = 1 / (1 + xn);
        fj3[n] = (1 - xn) / (1 + xn);
        fj3[grid_col-2-n] = (1 - xn) / (1 + xn);
    }
}

void creat_cylinders(pair<int, int> *centers, int num, double *gaz, double *gbz,
                     double radius, double epsr, double sigma, double delta_t,
                     double epsz){
    for(int n=0; n<num; n++){
        int x = centers[n].first;
        int y = centers[n].second;
        for(int i=0;i<grid_row;i++){
            for(int j=0; j<grid_col; j++){
                double dist = sqrt(pow((x-i),2) + pow((y-j),2));
                if (dist <= radius) {
                    gaz[i*grid_col+j] = 1 / (epsr + (sigma * delta_t / epsz));
                    gbz[i*grid_col+j] = (sigma * delta_t / epsz);
                }
            }
        }
    }
}

void calculateDz(double *dz, double *hx, double *hy, double *gi3, double *gj3,
                 double *gi2, double *gj2){
    for(int i=1; i<grid_row; i++){
        for(int j=1; j<grid_col; j++){
            dz[i*grid_col+j] = gi3[i] * gj3[j] * dz[i*grid_col+j] + gi2[i] * gj2[j] * 0.5 *
                    (hy[i*grid_col+j] - hy[(i-1)*grid_col+j] - hx[i*grid_col+j] + hx[i*grid_col+j-1]);
        }
    }
}

void calculateEz(double *ez, double *dz, double *gaz, double *gbz, double *iz){
    for(int i=0; i<grid_row; i++){
        for(int j=0; j<grid_col; j++){
            ez[i*grid_col+j] = gaz[i*grid_col+j] * (dz[i*grid_col+j] - iz[i*grid_col+j]);
            iz[i*grid_col+j] = iz[i*grid_col+j] + gbz[i*grid_col+j] * ez[i*grid_col+j];
        }
    }
}

void calculateHx(double *ez, double *ihx, double *hx, double *fi1, double *fj2, double *fj3){
    for(int i=0; i<grid_row; i++) {
        for(int j=0; j<grid_col-1; j++) {
            double curl_e = ez[i*grid_col+j] - ez[i*grid_col+j+1];
            ihx[i*grid_col+j] = ihx[i*grid_col+j] + curl_e;
            hx[i*grid_col+j] = fj3[j] * hx[i*grid_col+j] + fj2[j] * (0.5 * curl_e + fi1[i] * ihx[i*grid_col+j]);
        }
    }
}

void calculateHy(double *ez, double *hy, double *ihy, double *fi2, double *fi3, double *fj1){
    for (int i=0; i<grid_row-1; i++) {
        for (int j=0; j<grid_col; j++) {
            double curl_e = ez[i*grid_col+j] - ez[(i+1)*grid_col+j];
            ihy[i*grid_col+j] = ihy[i*grid_col+j] + curl_e;
            hy[i*grid_col+j] = fi3[i] * hy[i*grid_col+j] - fi2[i] * (0.5 * curl_e + fj1[j] * ihy[i*grid_col+j]);
        }
    }
}
