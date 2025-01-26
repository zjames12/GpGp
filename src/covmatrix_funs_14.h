#ifndef COVMATRIX_FUNS_14_H
#define COVMATRIX_FUNS_14_H



// covariance functions
#include <RcppArmadillo.h>
#include <iostream>
#include <vector>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>

//' Local anisotropic exponential covariance function
//'
//' From a matrix of locations and covariance parameters of the form
//' (variance, range, nugget), return the square matrix of
//' all pairwise covariances.
//' @param locs A matrix with \code{n} rows and \code{d} columns.
//' Each row of locs is a point in R^d.
//' @param covparms A vector with covariance parameters
//' in the form (variance, range, nugget)
//' @return A matrix with \code{n} rows and \code{n} columns, with the i,j entry
//' containing the covariance between observations at \code{locs[i,]} and
//' \code{locs[j,]}.
//' @section Parameterization:
//' The covariance parameter vector is (variance, range, nugget)
//' = \eqn{(\sigma^2,\alpha,\tau^2)}, and the covariance function is parameterized
//' as
//' \deqn{ M(x,y) = \sigma^2 exp( - || x - y ||/ \alpha )}
//' The nugget value \eqn{ \sigma^2 \tau^2 } is added to the diagonal of the covariance matrix.
//' NOTE: the nugget is \eqn{ \sigma^2 \tau^2 }, not \eqn{ \tau^2 }. 
// [[Rcpp::export]]
arma::mat exponential_local_anisotropic(arma::vec covparms, arma::mat locs, Rcpp::NumericVector arr){
    Rcpp::IntegerVector dims = arr.attr("dim");
    int n_rows = dims[0];
    int n_cols = dims[1];
    int n_slices = dims[2];
    arma::cube aniso(arr.begin(), n_rows, n_cols, n_slices, false);
    int dim = locs.n_cols;
    int n = locs.n_rows;
    double nugget = covparms( 0 )*covparms( 1 );
    // create scaled locations
    // mat locs_scaled(n,dim);
    // for(int j=0; j<dim; j++){ 
    //     for(int i=0; i<n; i++){
    //         locs_scaled(i,j) = locs(i,j)/covparms(1);
    //     }
    // }
    // calculate covariances
    arma::mat covmat(n,n);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        // calculate distance

        arma::rowvec diff = locs.row(i1) - locs.row(i2);
        // aniso.slice(i1).print();
        // printf("\n");
        // aniso.slice(i2).print();
        // printf("\n");
        arma::mat quad = diff * inv_sympd((aniso.slice(i1) + aniso.slice(i2)) / 2) * diff.t();
        double q = sqrt(quad(0,0));
        
        // double d = 0.0;
        // for(int j=0; j<dim; j++){
        //     d += pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
        // }
        // d = std::sqrt( d );

        // calculate covariance 
        if (i1 == i2) {           
            covmat(i2,i1) =  covparms(0)*std::exp( -q );
        } else {
            covmat(i2,i1) = pow(det(aniso.slice(i1)),.25) * pow(det(aniso.slice(i2)),.25) * pow(det((aniso.slice(i1)+aniso.slice(i2))/2),-.5);
            covmat(i2,i1) *= covparms(0)*std::exp( -q );
            covmat(i1,i2) = covmat(i2,i1);
        }

    }}
    // add nugget
    for(int i1=0; i1<n; i1++){
	covmat(i1,i1) += nugget;
    }
    return covmat;
}

//' @describeIn exponential_local_anisotropic Derivatives of local anisotropic exponential covariance
// [[Rcpp::export]]
arma::cube d_exponential_local_anisotropic(arma::vec covparms, arma::mat locs, Rcpp::NumericVector arr){

    int dim = locs.n_cols;
    int n = locs.n_rows;

    Rcpp::IntegerVector dims = arr.attr("dim");
    int n_rows = dims[0];
    int n_cols = dims[1];
    int n_slices = dims[2];
    arma::cube aniso(arr.begin(), n_rows, n_cols, n_slices, false);
    //double nugget = covparms( 0 )*covparms( 2 );
    // create scaled locations
    // mat locs_scaled(n,dim);
    // for(int j=0; j<dim; j++){ 
    //     for(int i=0; i<n; i++){
    //         locs_scaled(i,j) = locs(i,j)/covparms(1);
    //     }
    // }
    // calculate derivatives
    arma::cube dcovmat = arma::cube(n,n,covparms.n_elem, fill::zeros);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        // double d = 0.0;
        // for(int j=0; j<dim; j++){
        //     d += pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
        // }
        // d = std::sqrt( d );
        arma::rowvec diff = locs.row(i1) - locs.row(i2);
        arma::mat quad = diff * inv_sympd((aniso.slice(i1) + aniso.slice(i2)) / 2) * diff.t();
        double q = sqrt(quad(0,0));

        double c = pow(det(aniso.slice(i1)),.25) * pow(det(aniso.slice(i2)),.25) * pow(det((aniso.slice(i1)+aniso.slice(i2))/2),-.5);
        
        dcovmat(i1,i2,0) += c * std::exp(-q);
        // dcovmat(i1,i2,1) += covparms(0)*std::exp(-q)*q/covparms(1);
        if( i1 == i2 ){ // update diagonal entry
            dcovmat(i1,i2,0) += c * covparms(1);
            dcovmat(i1,i2,1) += c * covparms(0); 
        } else { // fill in opposite entry
            for(int j=0; j<covparms.n_elem; j++){
                dcovmat(i2,i1,j) = dcovmat(i1,i2,j);
            }
        }
    }}

    return dcovmat;
}

#endif