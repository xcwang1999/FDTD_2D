#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

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
__global__ void incident_Ez_values(double *ez_inc, double *hx_inc);
__global__ void absorbing_boundary_condition(double *ez_inc, double *boundary_low, double *boundary_high);
__global__ void inject_source(double *ez_inc, double pulse, int t0, int spread, int time_step);
__global__ void calculate_Dz(double *dz, double *hx, double *hy, double *gi3, double *gj3,
                             double *gi2, double *gj2);
__global__ void incident_Dz(double *dz, double *hx_inc, int ia, int ib, int ja, int jb);
__global__ void calculate_Ez(double *ez, double *dz, double *gaz, double *gbz, double *iz);
__global__ void calculate_incident_Hx(double *hx_inc, double *ez_inc);
__global__ void calculate_Hx(double *ez, double *ihx, double *hx, double *fi1, double *fj2, double *fj3);
__global__ void incident_Hx(double *hx, double *ez_inc, int ia, int ib, int ja, int jb);
__global__ void calculate_Hy(double *ez, double *hy, double *ihy, double *fi2, double *fi3, double *fj1);
__global__ void incident_Hy(double *hy, double *ez_inc, int ia, int ib, int ja, int jb);
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
    double *ez_device, *dz_device, *hx_device, *hy_device, *iz_device,
            *ihx_device, *ihy_device, *ez_inc_device, *hx_inc_device;
    double *gaz_device, *gbz_device;
    double *gi2_device, *gi3_device, *fi1_device, *fi2_device, *fi3_device,
            *gj2_device, *gj3_device, *fj1_device, *fj2_device, *fj3_device;
    double *boundary_low_device, *boundary_high_device;

    cudaMalloc((double **)&ez_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&dz_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&hx_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&hy_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&iz_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&ihx_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&ihy_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&ez_inc_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&hx_inc_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&gaz_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&gbz_device, sizeof(double)*grid_row*grid_col);
    cudaMalloc((double **)&gi2_device, sizeof(double)*grid_row);
    cudaMalloc((double **)&gi3_device, sizeof(double)*grid_row);
    cudaMalloc((double **)&fi1_device, sizeof(double)*grid_row);
    cudaMalloc((double **)&fi2_device, sizeof(double)*grid_row);
    cudaMalloc((double **)&fi3_device, sizeof(double)*grid_row);
    cudaMalloc((double **)&gj2_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&gj3_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&fj1_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&fj2_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&fj3_device, sizeof(double)*grid_col);
    cudaMalloc((double **)&boundary_low_device, sizeof(double)*2);
    cudaMalloc((double **)&boundary_high_device, sizeof(double)*2);

    cudaMemcpy(ez_device, ez, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(dz_device, dz, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(hx_device, hx, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(hy_device, hy, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(iz_device, iz, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(ihx_device, ihx, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(ihy_device, ihy, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(ez_inc_device, ez_inc, sizeof(double)*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(hx_inc_device, hx_inc, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(gaz_device, gaz, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(gbz_device, gbz, sizeof(double)*grid_row*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(gi2_device, gi2, sizeof(double)*grid_row, cudaMemcpyHostToDevice);
    cudaMemcpy(gi3_device, gi3, sizeof(double)*grid_row, cudaMemcpyHostToDevice);
    cudaMemcpy(fi1_device, fi1, sizeof(double)*grid_row, cudaMemcpyHostToDevice);
    cudaMemcpy(fi2_device, fi2, sizeof(double)*grid_row, cudaMemcpyHostToDevice);
    cudaMemcpy(fi3_device, fi3, sizeof(double)*grid_row, cudaMemcpyHostToDevice);
    cudaMemcpy(gj2_device, gj2, sizeof(double)*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(gj3_device, gj3, sizeof(double)*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(fj1_device, fj1, sizeof(double)*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(fj2_device, fj2, sizeof(double)*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(fj3_device, fj3, sizeof(double)*grid_col, cudaMemcpyHostToDevice);
    cudaMemcpy(boundary_low_device, boundary_low, sizeof(double)*2, cudaMemcpyHostToDevice);
    cudaMemcpy(boundary_high_device, boundary_high, sizeof(double)*2, cudaMemcpyHostToDevice);

    const int nsteps = 2000;
    dim3 block_size(128, 128);
    dim3 grid_size((grid_row-1)/block_size.x + 1, (grid_col-1)/block_size.y + 1);
    // Main FDTD loop
    for(int time_step=1; time_step<=nsteps; time_step++){

        incident_Ez_values<<<(grid_col-1)/128+128, 128>>>(ez_inc_device, hx_inc_device);

        absorbing_boundary_condition<<<1, 1>>>(ez_inc_device, boundary_low_device, boundary_high_device);

        calculate_Dz<<<block_size, grid_size>>>(dz_device, hx_device, hy_device, gi3_device, gj3_device, gi2_device, gj2_device);

        inject_source<<<1, 1>>>(ez_inc_device, pulse, t0, spread, time_step);

        incident_Dz<<<(grid_row-1)/128+128, 128>>>(dz_device, hx_inc_device, ia, ib, ja, jb);

        calculate_Ez<<<block_size, grid_size>>>(ez_device, dz_device, gaz_device, gbz_device, iz_device);

        calculate_incident_Hx<<<(grid_col-1) / 128 + 128, 128>>>(hx_inc_device, ez_inc_device);

        calculate_Hx<<<block_size, grid_size>>>(ez_device, ihx_device, hx_device, fi1_device, fj2_device, fj3_device);

        incident_Hx<<<(grid_row-1) / 128 + 128, 128>>>(hx_device, ez_inc_device, ia, ib, ja, jb);

        calculate_Hy<<<block_size, grid_size>>>(ez_device, hy_device, ihy_device, fi2_device, fi3_device, fj1_device);

        incident_Hy<<<(grid_col-1) / 128 + 128, 128>>>(hy_device, ez_inc_device, ia, ib, ja, jb);

        double *ez_host = (double *)malloc(sizeof(double)*grid_row*grid_col);
        cudaMemcpy(ez_host, ez_device, sizeof(double)*grid_row*grid_col, cudaMemcpyDeviceToHost);
        ofstream outfile;
        outfile.open("data_cu.txt", ios::app);
        for(int i=0; i<grid_col; i++){
            for(int j=0; j<grid_row; j++){
                if(j!=grid_row-1){
                    outfile << setprecision(4) << fixed << ez_host[i*grid_col+j] << " ";
                } else{
                    outfile << setprecision(4) << fixed << ez_host[i*grid_col+j] << endl;
                }
            }
        }
        outfile.close();
        free(ez_host);
    }
    cudaFree(ez_device);
    cudaFree(dz_device);
    cudaFree(hx_device);
    cudaFree(hy_device);
    cudaFree(iz_device);
    cudaFree(ihx_device);
    cudaFree(ihy_device);
    cudaFree(ez_inc_device);
    cudaFree(hx_inc_device);
    cudaFree(gaz_device);
    cudaFree(gbz_device);
    cudaFree(gi2_device);
    cudaFree(gi3_device);
    cudaFree(fi1_device);
    cudaFree(fi2_device);
    cudaFree(fi3_device);
    cudaFree(gj2_device);
    cudaFree(gj3_device);
    cudaFree(fj1_device);
    cudaFree(fj2_device);
    cudaFree(fj3_device);
    cudaFree(boundary_low_device);
    cudaFree(boundary_high_device);
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
    }
    for(int n=0; n<grid_row; n++){
        gi3[n] = 1;
    }
    for(int n=0; n<grid_row; n++){
        fi2[n] = 1;
    }
    for(int n=0; n<grid_row; n++){
        fi3[n] = 1;
    }
    for(int n=0; n<grid_col; n++){
        gj2[n] = 1;
    }
    for(int n=0; n<grid_col; n++){
        gj3[n] = 1;
    }
    for(int n=0; n<grid_col; n++){
        fj2[n] = 1;
    }
    for(int n=0; n<grid_col; n++){
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

__global__ void incident_Ez_values(double *ez_inc, double *hx_inc){
    int j = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    if(j < grid_col){
        ez_inc[j] = ez_inc[j] + 0.5 * (hx_inc[j-1] - hx_inc[j]);
    }
}

__global__ void absorbing_boundary_condition(double *ez_inc, double *boundary_low, double *boundary_high){
    ez_inc[0] = boundary_low[0];
    boundary_low[0]=boundary_low[1];
    boundary_low[1]=ez_inc[1];

    ez_inc[grid_col-1] = boundary_high[0];
    boundary_high[0]=boundary_high[1];
    boundary_high[1]=ez_inc[grid_col-2];
}

__global__ void calculate_Dz(double *dz, double *hx, double *hy, double *gi3, double *gj3,
                             double *gi2, double *gj2){
    int i = 1 + blockIdx.x*blockDim.x + threadIdx.x;
    int j = 1 + blockIdx.y*blockDim.y + threadIdx.y;
    if(i<grid_row && j<grid_col){
        dz[i*grid_col+j] = gi3[i] * gj3[j] * dz[i*grid_col+j] + gi2[i] * gj2[j] * 0.5 *
                    (hy[i*grid_col+j] - hy[(i-1)*grid_col+j] - hx[i*grid_col+j] + hx[i*grid_col+j-1]);

    }
}

__global__ void inject_source(double *ez_inc, double pulse, int t0, int spread, int time_step){
    pulse = exp(-0.5 * pow(((t0-time_step) / spread), 2));
    ez_inc[3] = pulse;
}

__global__ void incident_Dz(double *dz, double *hx_inc, int ia, int ib, int ja, int jb){
    int i = ia + blockIdx.x*blockDim.x + threadIdx.x;
    if(i <= ib){
        dz[i*grid_col+ja] = dz[i*grid_col+ja] + 0.5 * hx_inc[ja-1];
        dz[i*grid_col+jb] = dz[i*grid_col+jb] - 0.5 * hx_inc[jb];
    }
}

__global__ void calculate_Ez(double *ez, double *dz, double *gaz, double *gbz, double *iz){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<grid_row && j<grid_col){
        ez[i*grid_col+j] = gaz[i*grid_col+j] * (dz[i*grid_col+j] - iz[i*grid_col+j]);
        iz[i*grid_col+j] = iz[i*grid_col+j] + gbz[i*grid_col+j] * ez[i*grid_col+j];
    }
}

__global__ void calculate_incident_Hx(double *hx_inc, double *ez_inc){
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(j<grid_col-1){
        hx_inc[j] = hx_inc[j] + 0.5 * (ez_inc[j] - ez_inc[j+1]);
    }
}

__global__ void calculate_Hx(double *ez, double *ihx, double *hx, double *fi1, double *fj2, double *fj3){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<grid_row && j<(grid_col-1)){
        double curl_e = ez[i*grid_col+j] - ez[i*grid_col+j+1];
        ihx[i*grid_col+j] = ihx[i*grid_col+j] + curl_e;
        hx[i*grid_col+j] = fj3[j] * hx[i*grid_col+j] + fj2[j] * (0.5 * curl_e + fi1[i] * ihx[i*grid_col+j]);

    }
}

__global__ void incident_Hx(double *hx, double *ez_inc, int ia, int ib, int ja, int jb){
    int i = ia + blockIdx.x*blockDim.x + threadIdx.x;
    if(i <= ib){
        hx[i*grid_col+ja-1] = hx[i*grid_col+ja-1] + 0.5 * ez_inc[ja];
        hx[i*grid_col+jb] = hx[i*grid_col+jb] - 0.5 * ez_inc[jb];
    }

}

__global__ void calculate_Hy(double *ez, double *hy, double *ihy, double *fi2, double *fi3, double *fj1){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i<grid_row-1 &&  j<grid_col){
        double curl_e = ez[i*grid_col+j] - ez[(i+1)*grid_col+j];
        ihy[i*grid_col+j] = ihy[i*grid_col+j] + curl_e;
        hy[i*grid_col+j] = fi3[i] * hy[i*grid_col+j] - fi2[i] * (0.5 * curl_e + fj1[j] * ihy[i*grid_col+j]);

    }
}

__global__ void incident_Hy(double *hy, double *ez_inc, int ia, int ib, int ja, int jb){
    int j = ja + blockIdx.x*blockDim.x + threadIdx.x;
    if(j <= jb){
        hy[(ia-1)*grid_col+j] = hy[(ia-1)*grid_col+j] - 0.5 * ez_inc[j];
        hy[ib*grid_col+j] = hy[ib*grid_col+j] + 0.5 * ez_inc[j];
    }
}