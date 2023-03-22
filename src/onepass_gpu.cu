// #include <RcppArmadillo.h>
// #include <omp.h>
// #include "onepass.h"


//' GPU Error check function
//`
//' Kernels do not throw exceptions. They instead return exit codes. If the exit code is
//` not \code{cudaSuccess} an error message is printed and the code is aborted.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    /*printf(cudaGetErrorString(code));
    printf("\n");*/
    if (code != cudaSuccess)
    {
        // printf("fail%i\n", code);
        // fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);

        //if (abort) exit(code);
    }

}
// Kernel to calculate compute pieces
// 
// y n
// X n * p
// NNarray n * m
// locs n * dim
// 
// locs_scaled n * m * dim
// covmat n * m * m
// 
// logdet n                     // needs to be summed on axis 0 to form scalar result
// ySy n                        // needs to be summed on axis 0 to form scalar result
// XSX n * p * p                // needs to be summed on axis 0 to form p * p result
// ySX n * p                    // needs to be summed on axis 0 to form p result
// dXSX n * p * p * nparms      // needs to be summed on axis 0 to form p * p * nparms result
// dySX n * p * nparms          // needs to be summed on axis 0 to form p * nparms result
// dySy n * nparms              // needs to be summed on axis 0 to form nparms result
// dlogdet n * nparms           // needs to be summed on axis 0 to form nparms result
// ainfo n * nparms * nparms    // needs to be summed on axis 0 to form nparms * nparms result
// 
// dcovmat n * m * m * nparms
// ysub n * m
// X0 n * m * p
// Liy0 n * m
// LiX0 n * m * p
// choli2 n * m
// onevec m
// LidSLi2 n * m * nparms
// c n * m
// v1 n * p
// LidSLi3 n * m
//
__global__ void compute_pieces(double* y, double* X, double* NNarray, double* locs, double* locsub,
    double* covmat, double* logdet, double* ySy, double* XSX, double* ySX,
    double* dXSX , double* dySX, double* dySy, double* dlogdet, double* ainfo,
    double variance, double range, double nugget,
    int n, int p, int m, int dim, int nparms,
    bool profbeta, bool grad_info,
    double* dcovmat,
    double* ysub, double* X0, double* Liy0, double* LiX0, double* choli2, double* onevec,
    double* LidSLi2, double* c, double* v1, double* LidSLi3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int bsize = std::min(i + 1, m);
    clock_t full_start = clock();
    if (i >= m && i < n) {
        for (int j = m - 1; j >= 0; j--) {
            ysub[i * m + m - 1 - j] = y[static_cast<int>(NNarray[i * m + j]) - 1];
            for (int k = 0; k < dim; k++) {
                locsub[i * m * dim + (m - 1 - j) * dim + k] = locs[(static_cast<int>(NNarray[i * m + j]) - 1) * dim + k] / range;
            }
            if (profbeta) {
                for (int k = 0; k < p; k++) { 
                    X0[i * m * p + (m - 1 - j) * p + k] = X[(static_cast<int>(NNarray[i * m + j]) - 1) * p + k]; 
                }
            }
        }
       
        
        // compute covariance matrix and derivatives and take cholesky
        // arma::mat covmat = p_covfun[0](covparms, locsub);
        //double* covmat = exponential_isotropic(covparms, locsub, m, dim);
        // Calculate covmatrix
        double temp;
        for (int i1 = 0; i1 < m; i1++) {
            for (int i2 = 0; i2 <= i1; i2++) {
                double d = 0.0;
                for (int j = 0; j < dim; j++) {
                    temp = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                    d += temp * temp;
                }
                d = sqrt(d);
                // calculate covariance
                if (i1 == i2) {
                    covmat[i * m * m + i2 * m + i1] = variance * (exp(-d) + nugget);
                }
                else {
                    covmat[i * m * m + i2 * m + i1] = variance * exp(-d);
                    covmat[i * m * m + i1 * m + i2] = covmat[i * m * m + i2 * m + i1];
                }
            }
        }
        
        
        if (grad_info) {
            // calculate derivatives
            //arma::cube dcovmat = arma::cube(n, n, covparms.n_elem, fill::zeros);
            //dcovmat = (double*)malloc(sizeof(double) * m * m * nparms);
            for (int i1 = 0; i1 < m; i1++) {
                for (int i2 = 0; i2 <= i1; i2++) {
                    double d = 0.0;
                    double a = 0;
                    for (int j = 0; j < dim; j++) {
                        a = locsub[i * m * dim + i1 * dim + j] - locsub[i * m * dim + i2 * dim + j];
                        d += a * a;
                    }
                    d = sqrt(d);
                    temp = exp(-d);

                    dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += temp;
                    dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 1] += variance * temp * d / range;
                    if (i1 == i2) { // update diagonal entry
                        dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 0] += nugget;
                        dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + 2] = variance;
                    }
                    else { // fill in opposite entry
                        for (int j = 0; j < nparms; j++) {
                            dcovmat[i * m * m * nparms + i2 * m * nparms + i1 * nparms + j] = dcovmat[i * m * m * nparms + i1 * m * nparms + i2 * nparms + j];
                        }
                    }
                }
            }
            
        }
        
        /*arma::mat cholmat = eye(size(covmat));
        chol(cholmat, covmat, "lower");*/

        // Cholesky decomposition
        //int k, q, j;
        double temp2;
        double diff;
       
        int r, j, k;
        for (r = 0; r < m + 0; r++) {
            diff = 0;
            for (k = 0; k < r; k++) {
                temp = covmat[i * m * m + r * m + k];
                diff += temp * temp;
            }
            covmat[i * m * m + r * m + r] = sqrt(covmat[i * m * m + r * m + r] - diff);


            for (j = r + 1; j < m + 0; j++) {
                diff = 0;
                for (k = 0; k < r; k++) {
                    diff += covmat[i * m * m + r * m + k] * covmat[i * m * m + j * m + k];
                }
                covmat[i * m * m + j * m + r] = (covmat[i * m * m + j * m + r] - diff) / covmat[i * m * m + r * m + r];
            }
        }

       
        // i1 is conditioning set, i2 is response        
        //arma::span i1 = span(0,bsize-2);
        //arma::span i2 = span(bsize - 1, bsize - 1);

        // get last row of cholmat
        /*arma::vec onevec = zeros(bsize);
        onevec(bsize - 1) = 1.0;*/
        
        if (grad_info) {
            //choli2 = backward_solve(cholmat, onevec, m);
            choli2[i * m + m - 1] = 1 / covmat[i * m * m + (m - 1) * m + m - 1];

            for (int k = m - 2; k >= 0; k--) {
                double dd = 0.0;
                for (int j = m - 1; j > k; j--) {
                    dd += covmat[i * m * m + j * m + k] * choli2[i * m + j];
                }
                choli2[i * m + k] = ( - dd) / covmat[i * m * m + k * m + k];
            }
        }
       
        //bool cond = bsize > 1;

        // do solves with X and y
        if (profbeta) {
            // LiX0 = forward_solve_mat(cholmat, X0, m, p);
            for (int k = 0; k < p; k++) { 
                LiX0[i * m * p + 0 * p + k] = X0[i * m * p + 0 * p + k] / covmat[i * m * m + 0 * m + 0]; 
            }
            
            for (int h = 1; h < m; h++) {
                for (int k = 0; k < p; k++) {
                    double dd = 0.0;
                    for (int j = 0; j < h; j++) {
                        dd += covmat[i * m * m + h * m + j] * LiX0[i * m * p + j * p + k];
                    }
                    LiX0[i * m * p + h * p + k] = (X0[i * m * p + h * p + k] - dd) / covmat[i * m * m + h * m + h];
                }
            }
            
            
        }
        for (int j = 0; j < m; j++) {
            for (int k = j + 1; k < m; k++) {
                covmat[i * m * m + j * m + k] = 0.0;
            }
        }
        
        //arma::vec Liy0 = solve( trimatl(cholmat), ysub );
        //double* Liy0 = forward_solve(cholmat, ysub, m);
        //double* Liy0 = (double*)malloc(sizeof(double) * m);
        /*for (int j = 0; j < m; j++) {
            Liy0[i * m + j] = 0.0f;
        }*/
        Liy0[i * m + 0] = ysub[i * m + 0] / covmat[i * m * m + 0 * m + 0];
        
        for (int k = 1; k < m; k++) {
            double dd = 0.0;
            for (int j = 0; j < k; j++) {
                dd += covmat[i * m * m + k * m + j] * Liy0[i * m + j];
            }
            Liy0[i * m + k] = (ysub[i * m + k] - dd) / covmat[i * m * m + k * m + k];
        }
       
        
        // loglik objects
        logdet[i] = 2.0 * log(covmat[i * m * m + (m - 1) * m + m - 1]);
        
        temp = Liy0[i * m + m - 1];
        ySy[i] = temp * temp;
        
        
        if (profbeta) {
            /*l_XSX += LiX0.rows(i2).t() * LiX0.rows(i2);
            l_ySX += (Liy0(i2) * LiX0.rows(i2)).t();*/
            temp2 = Liy0[i * m + m - 1];
            for (int i1 = 0; i1 < p; i1++) {
                temp = LiX0[i * m * p + (m - 1) * p + i1];
                for (int i2 = 0; i2 <= i1; i2++) {
                    XSX[i * p * p + i1 * p + i2] = temp * LiX0[i * m * p + (m - 1) * p + i2];
                    XSX[i * p * p + i2 * p + i1] = XSX[i * p * p + i1 * p + i2];
                }
                ySX[i * p + i1] = temp2 * LiX0[i * m * p + (m - 1) * p + i1];
            }
            
        }
        if (grad_info) {
            // gradient objects
            // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
            // LidSLi2 stores these columns in a matrix for all parameters
            // arma::mat LidSLi2(bsize, nparms);
            
           
            
            for (int j = 0; j < nparms; j++) {
                // compute last column of Li * (dS_j) * Lit
                //arma::vec LidSLi3 = forward_solve(cholmat, dcovmat.slice(j) * choli2);
                // c = dcovmat.slice(j) * choli2
                for (int h = 0; h < m; h++) {
                    c[i * m + h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        temp += dcovmat[i * m * m * nparms + h * m * nparms + k * nparms + j] * choli2[i * m + k];
                    }
                    c[i * m + h] = temp;
                }
                    
                //LidSLi3 = forward_solve(cholmat, c);      
                LidSLi3[i * m + 0] = c[i * m + 0] / covmat[i * m * m + 0 * m + 0];

                for (int k = 1; k < m; k++) {
                    double dd = 0.0;
                    for (int l = 0; l < k; l++) {
                        dd += covmat[i * m * m + k * m + l] * LidSLi3[i * m + l];
                    }
                    LidSLi3[i * m + k] = (c[i * m + k] - dd) / covmat[i * m * m + k * m + k];
                }
                    
                ////////////////
                //arma::vec v1 = LiX0.t() * LidSLi3;

                for (int h = 0; h < p; h++) {
                    v1[i * p + h] = 0;
                    temp = 0;
                    for (int k = 0; k < m; k++) {
                        temp += LiX0[i * m * p + k * p + h] * LidSLi3[i * m + k];
                    }
                    v1[i * p + h] = temp;
                }
                    
                ////////////////

                //double s1 = as_scalar(Liy0.t() * LidSLi3);
                double s1 = 0;
                for (int h = 0; h < m; h++) {
                    s1 += Liy0[i * m + h] * LidSLi3[i * m + h];
                }
                    
                ////////////////

                /*(l_dXSX).slice(j) += v1 * LiX0.rows(i2) + (v1 * LiX0.rows(i2)).t() -
                    as_scalar(LidSLi3(i2)) * (LiX0.rows(i2).t() * LiX0.rows(i2));*/

                    //double* v1LiX0 = (double*)malloc(sizeof(double) * m * m);
                double temp3;
                double temp4 = LidSLi3[i * m + m - 1];
                for (int h = 0; h < p; h++) {
                    temp = v1[i * p + h];
                    temp2 = LiX0[i * m * p + (m - 1) * p + h];
                        
                    for (int k = 0; k < p; k++) {
                        temp3 = LiX0[i * m * p + (m - 1) * p + k];
                        dXSX[i * p * p * nparms + h * p * nparms + k * nparms + j] = temp * temp3 +
                            (v1[i * p + k] - temp4 * temp3) * temp2;
                    }
                }
                temp = Liy0[i * m + m - 1];
                ///////////////
                /*(l_dySy)(j) += as_scalar(2.0 * s1 * Liy0(i2) -
                    LidSLi3(i2) * Liy0(i2) * Liy0(i2));*/
                dySy[i * nparms + j] = (2 * s1 - temp4 * temp) * temp;
                    
                /*(l_dySX).col(j) += (s1 * LiX0.rows(i2) + (v1 * Liy0(i2)).t() -
                    as_scalar(LidSLi3(i2)) * LiX0.rows(i2) * as_scalar(Liy0(i2))).t();*/
                temp3 = LidSLi3[i * m + m - 1];
                for (int h = 0; h < p; h++) {
                    temp2 = LiX0[i * m * p + (m - 1) * p + h];
                    dySX[i * p * nparms + h * nparms + j] = s1 * temp2 +
                        v1[i * p + h] * temp - temp3 * temp2 * temp;
                }
                    
                //(l_dlogdet)(j) += as_scalar(LidSLi3(i2));
                dlogdet[i * nparms + j] = temp3;

                //LidSLi2.col(j) = LidSLi3;
                for (int h = 0; h < m; h++) {
                    LidSLi2[i * m * nparms + h * nparms + j] = LidSLi3[i * m + h];
                }
                /*if (i == 40 && j == 2) {
                    printf("CPU s1\n");
                    printf("%f", s1);
                }*/
                    
            }
           
            // fisher information object
            // bottom right corner gets double counted, so subtract it off
            for (int h = 0; h < nparms; h++) {
                temp2 = LidSLi2[i * m * nparms + (m - 1) * nparms + h];
                for (int j = 0; j < h + 1; j++) {
                    /*(l_ainfo)(h, j) +=
                        1.0 * accu(LidSLi2.col(h) % LidSLi2.col(j)) -
                        0.5 * accu(LidSLi2.rows(i2).col(j) %
                            LidSLi2.rows(i2).col(h));*/
                    double s = 0;
                    for (int l = 0; l < m; l++) {
                        s += LidSLi2[i * m * nparms + l * m + h] * LidSLi2[i * m * nparms + l * m + j];
                    }
                    ainfo[i * nparms * nparms + h * nparms + j] = s - 0.5 * LidSLi2[i * m * nparms + (m - 1) * nparms + j] * temp2;
                }
            }
            
        }
    }
   
}

extern "C"
void call_compute_pieces_gpu(
    double* covparms,
    // std::string covfun_name,
    double* locs,
    double* NNarray,
    double* y,
    double* X,
    double* XSX,
    double* ySX,
    double* ySy,
    double* logdet,
    double* dXSX,
    double* dySX,
    double* dySy,
    double* dlogdet,
    double* ainfo,
    int profbeta,
    int grad_info,
    int n,
    int m,
    int p,
    int nparms,
    int dim
) {
    //m++;

    double* d_locs;
    double* d_NNarray;
    double* d_y;
    double* d_X;

    gpuErrchk(cudaMalloc((void**)&d_locs, sizeof(double) * n * dim));
    gpuErrchk(cudaMalloc((void**)&d_NNarray, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_y, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_X, sizeof(double) * n * p));

    double* d_covmat;
    double* d_locs_scaled;
    double* d_ySX;
    double* d_XSX;
    double* d_ySy;
    double* d_logdet;
    double* d_dXSX;
    double* d_dySX;
    double* d_dySy;
    double* d_dlogdet;
    double* d_dainfo;

    double* d_dcovmat;
    double* d_ysub;
    double* d_X0;
    double* d_Liy0;
    double* d_LiX0;
    double* d_choli2;
    double* d_onevec;
    double* d_LidSLi2;
    double* d_c;
    double* d_v1;
    double* d_LidSLi3;

    gpuErrchk(cudaMalloc((void**)&d_covmat, sizeof(double) * n * m * m));
    gpuErrchk(cudaMalloc((void**)&d_locs_scaled, sizeof(double) * n * m * dim));
    gpuErrchk(cudaMalloc((void**)&d_ySX, sizeof(double) * n * p ));
    gpuErrchk(cudaMalloc((void**)&d_XSX, sizeof(double) * n * p * p));
    gpuErrchk(cudaMalloc((void**)&d_ySy, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_logdet, sizeof(double) * n));
    gpuErrchk(cudaMalloc((void**)&d_dXSX, sizeof(double) * n * p * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySX, sizeof(double) * n * p * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dySy, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dlogdet, sizeof(double) * n * nparms));
    gpuErrchk(cudaMalloc((void**)&d_dainfo, sizeof(double) * n * nparms * nparms));

    gpuErrchk(cudaMalloc((void**)&d_dcovmat, sizeof(double) * n * m * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_ysub, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_X0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_Liy0, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_LiX0, sizeof(double) * n * m * p));
    gpuErrchk(cudaMalloc((void**)&d_choli2, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_onevec, sizeof(double) * m));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi2, sizeof(double) * n * m * nparms));
    gpuErrchk(cudaMalloc((void**)&d_c, sizeof(double) * n * m));
    gpuErrchk(cudaMalloc((void**)&d_v1, sizeof(double) * n * p));
    gpuErrchk(cudaMalloc((void**)&d_LidSLi3, sizeof(double) * n * m));

    gpuErrchk(cudaMemcpy(d_locs, locs, sizeof(double) * n * dim, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_NNarray, NNarray, sizeof(double) * n * m, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, sizeof(double) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_X, X, sizeof(double) * n * p, cudaMemcpyHostToDevice));

    int grid_size = 64;
    int block_size = ((n + grid_size) / grid_size);
    
    compute_pieces << <block_size, grid_size >> > (d_y, d_X, d_NNarray, d_locs, d_locs_scaled,
        d_covmat, d_logdet, d_ySy, d_XSX, d_ySX,
        d_dXSX, d_dySX, d_dySy, d_dlogdet, d_dainfo,
        covparms[0], covparms[1], covparms[2],
        n, p, m, dim, nparms,
        profbeta, grad_info,
        d_dcovmat,
        d_ysub, d_X0, d_Liy0, d_LiX0, d_choli2, d_onevec,
        d_LidSLi2, d_c, d_v1, d_LidSLi3);
    cudaDeviceSynchronize();


    double* l_ySy = (double*)malloc(sizeof(double) * n);
    double* l_logdet = (double*)malloc(sizeof(double) * n);
    double* l_ySX = (double*)malloc(sizeof(double) * n * p);
    double* l_XSX = (double*)malloc(sizeof(double) * n * p * p);
    double* l_dySX = (double*)malloc(sizeof(double) * n * p * nparms);
    double* l_dXSX = (double*)malloc(sizeof(double) * n * p * p * nparms);


    gpuErrchk(cudaMemcpy(l_ySy, d_ySy, sizeof(double) * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_logdet, d_logdet, sizeof(double) * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_ySX, d_ySX, sizeof(double) * n * p, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_XSX, d_XSX, sizeof(double) * n * p * p, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_dySX, d_dySX, sizeof(double) * n * p * nparms, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(l_dXSX, d_dXSX, sizeof(double) * n * p * p * nparms, cudaMemcpyDeviceToHost));
    
    /// Testing
    /*double* l_Liy0 = (double*)malloc(sizeof(double) * n * m);
    gpuErrchk(cudaMemcpy(l_Liy0, d_Liy0, sizeof(double) * n * m, cudaMemcpyDeviceToHost));
    
    printf("CPU Liy0\n");
    for (int j = 0; j < 10; j++) {
        printf("%f ", l_Liy0[40 * m + j]);
    }
    printf("\n");*/
    //////////////////



    ySy[0] = 0;
    logdet[0] = 0;
    for (int i = 0; i < n; i++) {
        /*if (i < 50) {
            printf("%f\n", l_ySy[i]);
        }*/
        ySy[0] += l_ySy[i];
        logdet[0] += l_logdet[i];
        for (int j = 0; j < p; j++) {
            ySX[j] += l_ySX[i * p + j];
            for (int k = 0; k < p; k++) {
                XSX[j * p + k] += l_XSX[i * p * p + j * p + k];
                for (int l = 0; l < nparms; l++) {
                    dXSX[j * p * nparms + k * nparms + l] += l_dXSX[i * p * p * nparms + j * p * nparms + k * nparms + l];
                    //dXSX[j * p * nparms + k * nparms + l] += 0;
                }
            }
            for (int k = 0; k < nparms; k++) {
                dySX[j * nparms + k] += l_dySX[i * p * nparms + j * nparms + k];
                //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
            }
        }
        //printf("%f\n", l_logdet[i]);
    }
    //printf("m:%i\n", m);
    // arma::vec covparmsa(covparms, 3);
    // arma::mat locsa(locs, dim, n);
    // arma::mat NNarraya(NNarray, m, m);
    // arma::vec ya(y, m);
    // arma::mat Xa(X, p, m);

    // locsa = locsa.t();
    // NNarraya = NNarraya.t();
    // Xa = Xa.t();

    // arma::mat XSXa = arma::mat(p, p, fill::zeros);
    // arma::vec ySXa = arma::vec(p, fill::zeros);
    // double ySya = 0.0;
    // double logdeta = 0.0;

    // // gradient objects    
    // arma::cube dXSXa = arma::cube(p, p, nparms, fill::zeros);
    // arma::mat dySXa = arma::mat(p, nparms, fill::zeros);
    // arma::vec dySya = arma::vec(nparms, fill::zeros);
    // arma::vec dlogdeta = arma::vec(nparms, fill::zeros);
    // // fisher information
    // arma::mat ainfoa = arma::mat(nparms, nparms, fill::zeros);

   
    // compute_pieces(
    //     covparmsa, "exponential_isotropic", locsa, NNarraya, ya, Xa,
    //     &XSXa, &ySXa, &ySya, &logdeta, &dXSXa, &dySXa, &dySya, &dlogdeta, &ainfoa,
    //     profbeta, grad_info
    // );
   
    // //printf("logdet: %f\n", logdet[0]);
        
    // ySy[0] += ySya;
    // logdet[0] += logdeta;
    // for (int j = 0; j < p; j++) {
    //     ySX[j] += ySXa(j);
    //     for (int k = 0; k < p; k++) {
    //         XSX[j * p + k] += XSXa(j, k);
    //         for (int l = 0; l < nparms; l++) {
    //             dXSX[j * p * nparms + k * nparms + l] += dXSXa(j, k, l);
    //             //dXSX[j * p * nparms + k * nparms + l] += 0;
    //         }
    //     }
    //     for (int k = 0; k < nparms; k++) {
    //         dySX[j * nparms + k] += dySXa(j, k);
    //         //printf("%f ", l_dySX[i * p * nparms + j * nparms + k]);
    //     }
    // }
    // //printf("%f\n", l_logdet[i]);
    
}

extern "C"
double return3(){
    return 3.0;
}

