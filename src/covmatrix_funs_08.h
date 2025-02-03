
#ifndef COVMATRIX_FUNS_nonstatvar_H
#define COVMATRIX_FUNS_nonstatvar_H

// covariance functions
#include <RcppArmadillo.h>
#include <iostream>
#include <vector>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace Rcpp;
using namespace arma;
//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::depends(BH)]]


//' Isotropic Matern covariance function, nonstationary variances
//'
//' From a matrix of locations and covariance parameters of the form
//' (variance, range, smoothness, nugget, <nonstat variance parameters>), 
//' return the square matrix of all pairwise covariances.
//' @param Z A matrix with \code{n} rows and \code{2} columns for spatial
//' locations + \code{p} columns describing spatial basis functions.
//' Each row of locs gives a point in R^2 (two dimensions only!) + the value
//' of \code{p} spatial basis functions.
//' @param covparms A vector with covariance parameters
//' in the form (variance, range, smoothness, nugget, <nonstat variance parameters>).
//' The number of nonstationary variance parameters should equal \code{p}.
//' @return A matrix with \code{n} rows and \code{n} columns, with the i,j entry
//' containing the covariance between observations at \code{locs[i,]} and
//' \code{locs[j,]}.
//' @section Parameterization:
//' This covariance function multiplies the isotropic Matern covariance
//' by a nonstationary variance function. The form of the covariance is
//' \deqn{ C(x,y) = exp( \phi(x) + \phi(y) ) M(x,y) }
//' where M(x,y) is the isotropic Matern covariance, and 
//' \deqn{ \phi(x) = c_1 \phi_1(x) + ... + c_p \phi_p(x) }
//' where \eqn{\phi_1,...,\phi_p} are the spatial basis functions
//' contained in the last \code{p} columns of \code{Z}, and 
//' \eqn{c_1,...,c_p} are the nonstationary variance parameters.
// [[Rcpp::export]]
arma::mat matern_nonstat_var(arma::vec covparms, arma::mat Z ){
    
    // this is a 2D covariance function!
    // first two columns of Z are spatial locations
    // rest of columns are values of basis functions at each location
    // log variance is linear in the basis functions
    // covparms(0) = overall variance
    // covparms(1) = isotropic range
    // covparms(2) = smoothness
    // covparms(3) = nugget
    // covparms(4) ... covparms(covparms.n_elem-1) = coefficients
    // in log linear variance function multiplying Z(,2) .... Z(,Z.n_cols-1)
    int dim = 2;
    int n = Z.n_rows;
    int nbasis = Z.n_cols - dim;
    int nisoparm = 4;
    double nugget = covparms( 0 )*covparms( 3 );
    double normcon = covparms(0)/(pow(2.0,covparms(2)-1.0)*boost::math::tgamma(covparms(2)) );
    
    // calculate covariances
    arma::mat covmat(n,n);
    for(int i1 = 0; i1 < n; i1++){
        for(int i2 = 0; i2 <= i1; i2++){
            
            // calculate scaled distance
            double d = 0.0;
            for(int j=0; j<dim; j++){
                d += pow( (Z(i1,j) - Z(i2,j))/covparms(1), 2.0 );
            }
            d = pow( d, 0.5 );
            
            // calculate nonstationary variance
            double v = 0.0;
            for(int j=0; j<nbasis; j++){
                v += ( Z(i1, j+dim) + Z(i2, j+dim) ) * covparms( j + nisoparm );
            }
            v = std::exp(v);
            
            if( d == 0.0 ){
                covmat(i2,i1) = covparms(0) * v;
            } else {
                // calculate covariance
                covmat(i2,i1) = normcon*v *
                    pow( d, covparms(2) ) * boost::math::cyl_bessel_k(covparms(2), d);
            }
            // add nugget
            if( i1 == i2 ){ covmat(i2,i2) += nugget; } 
            // fill in opposite entry
            else { covmat(i1,i2) = covmat(i2,i1); }
        }    
    }
    return covmat;
}

//' @describeIn matern_nonstat_var Derivatives with respect to parameters
// [[Rcpp::export]]
arma::cube d_matern_nonstat_var(arma::vec covparms, arma::mat Z ){

    int dim = 2;
    int n = Z.n_rows;
    int nbasis = Z.n_cols - dim;
    int nisoparm = 4;
    //double nugget = covparms( 0 )*covparms( 3 );
    double normcon = covparms(0)/(pow(2.0,covparms(2)-1.0)*boost::math::tgamma(covparms(2)));
    double eps = 1e-8;
    double normconeps = 
        covparms(0)/(pow(2.0,covparms(2)+eps-1.0)*boost::math::tgamma(covparms(2) + eps));
    
    // calculate derivatives
    arma::cube dcovmat = arma::cube(n,n,covparms.n_elem, fill::zeros);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        // calculate scaled distance
        double d = 0.0;
        for(int j=0; j<dim; j++){
            d += pow( (Z(i1,j) - Z(i2,j))/covparms(1), 2.0 );
        }
        d = pow( d, 0.5 );
        
        // calculate nonstationary variance
        double v = 0.0;
        for(int j=0; j<nbasis; j++){
            v += ( Z(i1, j+dim) + Z(i2, j+dim) ) * covparms( j + nisoparm );
        }
        v = std::exp(v);
        
        double cov;        
        if( d == 0.0 ){
            cov = covparms(0) * v;
            dcovmat(i2,i1,0) += v;
            for(int j=0; j<nbasis; j++){
                dcovmat(i2,i1,j+nisoparm) = cov*( Z(i1,j+dim) + Z(i2,j+dim) );
            }
        } else {
            cov = normcon * v *
                pow( d, covparms(2) ) *boost::math::cyl_bessel_k(covparms(2), d);
            // variance parameter
            dcovmat(i2,i1,0) += cov/covparms(0);
            // range parameter 
            dcovmat(i2,i1,1) += normcon * v * pow(d,covparms(2))*
                boost::math::cyl_bessel_k(covparms(2)-1.0, d)*d/covparms(1);
            // smoothness parameter (finite differencing)
            dcovmat(i2,i1,2) += 
                ( normconeps*v*pow(d,covparms(2)+eps)*
                boost::math::cyl_bessel_k(covparms(2)+eps, d)- cov )/eps;
            // log linear variance parameters
            for(int j=0; j<nbasis; j++){
                dcovmat(i2,i1,j+nisoparm) = cov*( Z(i1,j+dim) + Z(i2,j+dim) );
            }
        }
        if( i1 == i2 ){ // update diagonal entry
            dcovmat(i1,i2,0) += covparms(3);
            dcovmat(i1,i2,3) += covparms(0); 
        } else { // fill in opposite entry
            for(int j=0; j<covparms.n_elem; j++){
                dcovmat(i1,i2,j) = dcovmat(i2,i1,j);
            }
        }
    }}

    return dcovmat;
}






//' Isotropic exponential covariance function, nonstationary variances
//'
//' From a matrix of locations and covariance parameters of the form
//' (variance, range, nugget, <nonstat variance parameters>), 
//' return the square matrix of all pairwise covariances.
//' @param Z A matrix with \code{n} rows and \code{2} columns for spatial
//' locations + \code{p} columns describing spatial basis functions.
//' Each row of locs gives a point in R^2 (two dimensions only!) + the value
//' of \code{p} spatial basis functions.
//' @param covparms A vector with covariance parameters
//' in the form (variance, range, nugget, <nonstat variance parameters>).
//' The number of nonstationary variance parameters should equal \code{p}.
//' @return A matrix with \code{n} rows and \code{n} columns, with the i,j entry
//' containing the covariance between observations at \code{locs[i,]} and
//' \code{locs[j,]}.
//' @section Parameterization:
//' This covariance function multiplies the isotropic exponential covariance
//' by a nonstationary variance function. The form of the covariance is
//' \deqn{ C(x,y) = exp( \phi(x) + \phi(y) ) M(x,y) }
//' where M(x,y) is the isotropic exponential covariance, and 
//' \deqn{ \phi(x) = c_1 \phi_1(x) + ... + c_p \phi_p(x) }
//' where \eqn{\phi_1,...,\phi_p} are the spatial basis functions
//' contained in the last \code{p} columns of \code{Z}, and 
//' \eqn{c_1,...,c_p} are the nonstationary variance parameters.
// [[Rcpp::export]]
arma::mat exponential_nonstat_var(arma::vec covparms, arma::mat Z ){
    
    // this is a 2D covariance function!
    // first two columns of Z are spatial locations
    // rest of columns are values of basis functions at each location
    // log variance is linear in the basis functions
    // covparms(0) = overall variance
    // covparms(1) = isotropic range
    // covparms(2) = nugget
    // covparms(3) ... covparms(covparms.n_elem-1) = coefficients
    // in log linear variance function multiplying Z(,2) .... Z(,Z.n_cols-1)
    int dim = 2;
    int n = Z.n_rows;
    int nbasis = Z.n_cols - dim;
    int nisoparm = 3;
    double nugget = covparms( 0 )*covparms( 2 );

    // calculate covariances
    arma::mat covmat(n,n);
    for(int i1 = 0; i1 < n; i1++){
        for(int i2 = 0; i2 <= i1; i2++){
            
            // calculate scaled distance
            double d = 0.0;
            for(int j=0; j<dim; j++){
                d += pow( (Z(i1,j) - Z(i2,j))/covparms(1), 2.0 );
            }
            d = pow( d, 0.5 );
            
            // calculate nonstationary variance
            double v = 0.0;
            for(int j=0; j<nbasis; j++){
                v += ( Z(i1, j+dim) + Z(i2, j+dim) ) * covparms( j + nisoparm );
            }
            v = std::exp(v);
            
            if( d == 0.0 ){
                covmat(i2,i1) = covparms(0) * v;
            } else {
                // calculate covariance
                covmat(i2,i1) = covparms(0) * v * exp(-d);
            }
            // add nugget
            if( i1 == i2 ){ covmat(i2,i2) += nugget; } 
            // fill in opposite entry
            else { covmat(i1,i2) = covmat(i2,i1); }
        }    
    }
    return covmat;
}

//' @describeIn exponential_nonstat_var Derivatives with respect to parameters
// [[Rcpp::export]]
arma::cube d_exponential_nonstat_var(arma::vec covparms, arma::mat Z ){

    int dim = 2;
    int n = Z.n_rows;
    int nbasis = Z.n_cols - dim;
    int nisoparm = 3;
    //double nugget = covparms( 0 )*covparms( 2 );

    // calculate derivatives
    arma::cube dcovmat = arma::cube(n,n,covparms.n_elem, fill::zeros);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        // calculate scaled distance
        double d = 0.0;
        for(int j=0; j<dim; j++){
            d += pow( (Z(i1,j) - Z(i2,j))/covparms(1), 2.0 );
        }
        d = pow( d, 0.5 );
        
        // calculate nonstationary variance
        double v = 0.0;
        for(int j=0; j<nbasis; j++){
            v += ( Z(i1, j+dim) + Z(i2, j+dim) ) * covparms( j + nisoparm );
        }
        v = std::exp(v);
        
        double cov;        
        if( d == 0.0 ){
            cov = covparms(0) * v;
            dcovmat(i2,i1,0) += v;
            for(int j=0; j<nbasis; j++){
                dcovmat(i2,i1,j+nisoparm) = cov*( Z(i1,j+dim) + Z(i2,j+dim) );
            }
        } else {
            cov = covparms(0) * v * exp(-d);
            // variance parameter
            dcovmat(i2,i1,0) += cov/covparms(0);
            // range parameter
            dcovmat(i2,i1,1) += covparms(0) * v * exp(-d) * d / covparms(1);
            // log linear variance parameters
            for(int j=0; j<nbasis; j++){
                dcovmat(i2,i1,j+nisoparm) = cov*( Z(i1,j+dim) + Z(i2,j+dim) );
            }
        }
        if( i1 == i2 ){ // update diagonal entry
            dcovmat(i1,i2,0) += covparms(2);
            dcovmat(i1,i2,2) += covparms(0); 
        } else { // fill in opposite entry
            for(int j=0; j<covparms.n_elem; j++){
                dcovmat(i1,i2,j) = dcovmat(i2,i1,j);
            }
        }
    }}

    return dcovmat;
}

//' Isotropic exponential covariance function, nonstationary variances
//'
//' From a matrix of locations and covariance parameters of the form
//' (variance, range, nugget, <nonstat variance parameters>), 
//' return the square matrix of all pairwise covariances.
//' @param Z A matrix with \code{n} rows and \code{2} columns for spatial
//' locations + \code{p} columns describing spatial basis functions.
//' Each row of locs gives a point in R^2 (two dimensions only!) + the value
//' of \code{p} spatial basis functions.
//' @param covparms A vector with covariance parameters
//' in the form (variance, range, nugget, <nonstat variance parameters>).
//' The number of nonstationary variance parameters should equal \code{p}.
//' @return A matrix with \code{n} rows and \code{n} columns, with the i,j entry
//' containing the covariance between observations at \code{locs[i,]} and
//' \code{locs[j,]}.
//' @section Parameterization:
//' This covariance function multiplies the isotropic exponential covariance
//' by a nonstationary variance function. The form of the covariance is
//' \deqn{ C(x,y) = exp( \phi(x) + \phi(y) ) M(x,y) }
//' where M(x,y) is the isotropic exponential covariance, and 
//' \deqn{ \phi(x) = c_1 \phi_1(x) + ... + c_p \phi_p(x) }
//' where \eqn{\phi_1,...,\phi_p} are the spatial basis functions
//' contained in the last \code{p} columns of \code{Z}, and 
//' \eqn{c_1,...,c_p} are the nonstationary variance parameters.
// [[Rcpp::export]]
arma::mat exponential_nonstat_anisotropy(arma::vec covparms, arma::mat Z ){
    
    // this is a 2D covariance function!
    // first two columns of Z are spatial locations
    // rest of columns are values of basis functions at each location
    // log variance is linear in the basis functions
    // covparms(0) = overall variance
    // covparms(1) = isotropic range
    // covparms(2) = nugget
    // covparms(3) ... covparms(covparms.n_elem-1) = coefficients
    // in log linear variance function multiplying Z(,2) .... Z(,Z.n_cols-1)
    int dim = 2;
    int n = Z.n_rows;
    int nbasis = Z.n_cols - dim;
    int nisoparm = 2;
    double nugget = covparms( 0 )*covparms( 1 );

    // calculate covariances
    arma::mat covmat(n,n);
    for(int i1 = 0; i1 < n; i1++){
        for(int i2 = 0; i2 <= i1; i2++){
            // calculate nonstationary L11
            double L11_i1 = 0.0;
            double L11_i2 = 0.0;
            for(int j=0; j<nbasis; j++){
                L11_i1 += Z(i1, j+dim) * covparms( j + nisoparm );
                L11_i2 += Z(i2, j+dim) * covparms( j + nisoparm );
            }
            L11_i1 = std::exp(L11_i1);
            L11_i2 = std::exp(L11_i2);

            // calculate nonstationary L22
            double L22_i1 = 0.0;
            double L22_i2 = 0.0;
            for(int j=0; j<nbasis; j++){
                L22_i1 += Z(i1, j+dim) * covparms( j + nbasis + nisoparm );
                L22_i2 += Z(i2, j+dim) * covparms( j + nbasis + nisoparm );
            }
            L22_i1 = std::exp(L22_i1);
            L22_i2 = std::exp(L22_i2);

            // calculate nonstationary L12
            double L12_i1 = 0.0;
            double L12_i2 = 0.0;
            for(int j=0; j<nbasis; j++){
                L12_i1 += Z(i1, j+dim) * covparms( j + 2 * nbasis + nisoparm );
                L12_i2 += Z(i2, j+dim) * covparms( j + 2 * nbasis + nisoparm );
            }
            arma::mat L_i1 = {{L11_i1, 0}, {L12_i1, L22_i1}};
            arma::mat L_i2 = {{L11_i2, 0}, {L12_i2, L22_i2}};

            L_i2 = L_i2 * L_i2.t();
            L_i1 = L_i1 * L_i1.t();


            // calculate distance
            // double d = 0.0;
            // for(int j=0; j<dim; j++){
            //     d += pow( (Z(i1,j) - Z(i2,j)), 2.0 );
            // }
            // d = pow( d, 0.5 );
            arma::rowvec diff = Z.row(i1).subvec(0,dim-1) - Z.row(i2).subvec(0,dim-1);
            arma::mat quad = diff * inv_sympd((L_i1 + L_i2) / 2) * diff.t();
            double d = sqrt(quad(0,0));
            
            if( d == 0.0 ){
                covmat(i2,i1) = covparms(0);
            } else {
                // calculate covariance
                covmat(i2,i1) = pow(det(L_i1),.25) * pow(det(L_i2),.25) * pow(det((L_i1+L_i2)/2),-.5);
                covmat(i2,i1) *= covparms(0)*std::exp( -d );
            }
            // add nugget
            if( i1 == i2 ){ covmat(i2,i2) += nugget; } 
            // fill in opposite entry
            else { covmat(i1,i2) = covmat(i2,i1); }
        }    
    }
    return covmat;
}

// [[Rcpp::export]]
arma::cube d_exponential_nonstat_anisotropy(arma::vec covparms, arma::mat Z ){
    int dim = 2;
    int n = Z.n_rows;
    int nbasis = Z.n_cols - dim;
    int nisoparm = 2;
    //double nugget = covparms( 0 )*covparms( 2 );

    // calculate derivatives
    arma::cube dcovmat = arma::cube(n,n,covparms.n_elem, fill::zeros);
    for(int i1=0; i1<n; i1++){ for(int i2=0; i2<=i1; i2++){
        if (i1 == i2) {
            dcovmat(i1,i2,0) = 1+covparms(1);
            dcovmat(i1,i2,1) = covparms(0);
            for (int j = 0; j < nbasis; j++) {
                dcovmat(i1,i2,j+nisoparm) = 0;
            }
        } else {
        
            // calculate nonstationary L11
            double L11_i1 = 0.0;
            double L11_i2 = 0.0;
            for(int j=0; j<nbasis; j++){
                L11_i1 += Z(i1, j+dim) * covparms( j + nisoparm );
                L11_i2 += Z(i2, j+dim) * covparms( j + nisoparm );
            }
            L11_i1 = std::exp(L11_i1);
            L11_i2 = std::exp(L11_i2);

            // calculate nonstationary L22
            double L22_i1 = 0.0;
            double L22_i2 = 0.0;
            for(int j=0; j<nbasis; j++){
                L22_i1 += Z(i1, j+dim) * covparms( j + nbasis + nisoparm );
                L22_i2 += Z(i2, j+dim) * covparms( j + nbasis + nisoparm );
            }
            L22_i1 = std::exp(L22_i1);
            L22_i2 = std::exp(L22_i2);

            // calculate nonstationary L12
            double L12_i1 = 0.0;
            double L12_i2 = 0.0;
            for(int j=0; j<nbasis; j++){
                L12_i1 += Z(i1, j+dim) * covparms( j + 2 * nbasis + nisoparm );
                L12_i2 += Z(i2, j+dim) * covparms( j + 2 * nbasis + nisoparm );
            }
            arma::mat L_i1 = {{L11_i1, 0}, {L12_i1, L22_i1}};
            arma::mat L_i2 = {{L11_i2, 0}, {L12_i2, L22_i2}};

            arma::mat Sigma_i2 = L_i2 * L_i2.t();
            arma::mat Sigma_i1 = L_i1 * L_i1.t();

            arma::rowvec diff = Z.row(i1).subvec(0,dim-1) - Z.row(i2).subvec(0,dim-1);
            arma::mat quad = diff * inv_sympd((Sigma_i1 + Sigma_i2) / 2) * diff.t();
            double d = sqrt(quad(0,0));

            double det_Sigma_i1 = pow(det(Sigma_i1), .25);
            double det_Sigma_i2 = pow(det(Sigma_i2), .25);

            // Derivative for variance parameter
            dcovmat(i1,i2,0) = det_Sigma_i1 * det_Sigma_i2 * pow(det((Sigma_i1+Sigma_i2)/2),-.5)*std::exp( -d );
            dcovmat(i2,i1,0) = dcovmat(i1,i2,0);

            // Derivatives for covparms(2,..,2+nbasis) for L11
            for (int j = 0; j < nbasis; j++) {
                double d_detSigmai14 = 0.5 * pow(det(Sigma_i1), .25) * Z(i1, j + dim);
                double d_detSigmaj14 = 0.5 * pow(det(Sigma_i2), .25) * Z(i2, j + dim);

                double d_detSigma = pow(L11_i1*L22_i1,2)*Z(i1, j + dim) + pow(L11_i1*L22_i2,2)*Z(i1, j + dim) +
                                    pow(L11_i1,2)*Z(i1, j + dim) * pow(L12_i2, 2) + pow(L22_i1*L11_i2,2)*Z(i2, j + dim) +
                                    pow(L11_i2*L22_i2,2)*Z(i2, j + dim) + L11_i2*pow(L12_i1,2)* Z(i2, j + dim) -
                                    L11_i1*L11_i2*L12_i1*L12_i2*(Z(i1, j + dim) + Z(i2, j + dim));

                d_detSigma *= 0.5;
                double d_RQ  = pow(L12_i1,2)*pow(diff(1),2)*Z(i1, j + dim) - L11_i1*diff(0)*diff(1)*L12_i1*Z(i1,j+dim)+
                                pow(L11_i2,2)*pow(diff(1),2)*Z(i2, j + dim) - L11_i2*diff(0)*diff(1)*L12_i2*Z(i2,j+dim);
                d_RQ *= 2;
                d_RQ /= (pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)+pow(L22_i2,2)) + pow(L11_i2,2)*pow(L12_i1,2)-2*L11_i1*L11_i2*L12_i1*L12_i2+
                        pow(L11_i1,2)*pow(L12_i2,2);

                double temp = pow(L22_i1,2)*pow(diff(0),2)+pow(L22_i2,2)*pow(diff(0),2)+pow(L11_i1,2)*pow(diff(1),2)+pow(L11_i2,2)*pow(diff(1),2)-
                                2*L11_i1*diff(0)*diff(1)*L12_i1+pow(diff(0),2) + pow(diff(0),2)*pow(L12_i1,2)-
                                2*L11_i2*diff(0)*diff(1)*L12_i2+pow(diff(0),2)*pow(L12_i2,2);
                temp *= 2;
                temp *= 2*L11_i1*Z(i1,j+dim)*pow(L12_i2,2)+2*pow(L11_i2,2)*pow(L12_i1,2)*Z(i2,j+dim)-
                        2*L11_i1*L11_i2*L12_i1*L12_i2*(Z(i1,j+dim)+Z(i2,j+dim))+
                        (pow(L22_i1,2)+pow(L22_i2,2))*(2*pow(L11_i1,2)*Z(i1,j+dim)+2*pow(L11_i2,2)*Z(i2,j+dim));
                temp /= pow(((pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)+pow(L22_i2,2)) + pow(L11_i2,2)*pow(L12_i1,2)-2*L11_i1*L11_i2*L12_i1*L12_i2+
                        pow(L11_i1,2)*pow(L12_i2,2)),2);
                d_RQ += temp;

                dcovmat(i1,i2,j+nisoparm) = exp(-sqrt(quad(0,0)))/sqrt(det((Sigma_i1+Sigma_i2)/2)) * 
                    (det_Sigma_i1*d_detSigmai14 + det_Sigma_i2*d_detSigmaj14 + det_Sigma_i1*det_Sigma_i2*d_detSigma/(2*det((Sigma_i1+Sigma_i2)/2)) -
                    det_Sigma_i1*det_Sigma_i2*d_RQ/(2*sqrt(quad(0,0))));
                dcovmat(i2,i1,j+nisoparm) = dcovmat(i1,i2,j+nisoparm);
            }

            // Derivatives for covparms(nbasis + nisoparm,..,2 * nbasis + nisoparm) for L12
            for (int j = 0; j < nbasis; j++) {

                double d_detSigma = pow(L11_i2,2)*L12_i1*Z(i1,j + dim)-L11_i1*L11_i2*L12_i2*Z(i1,j + dim)-
                                    L11_i1*L11_i2*L12_i1*Z(i2,j + dim)+pow(L11_i1,2)*L12_i2*Z(i2,j + dim);
                d_detSigma *= 0.5;

                double d_RQ = pow(L22_i1,2)*pow(diff(0),2)+pow(L22_i2,2)*pow(diff(0),2)+pow(L11_i1,2)*pow(diff(1),2)+pow(L11_i2,2)*pow(diff(1),2)-
                                2*L11_i1*diff(0)*diff(1)*L12_i1+pow(diff(0),2)*pow(L12_i1,2)-2*L11_i2*diff(0)*diff(1)*L12_i2+pow(diff(0),2)*pow(L12_i2,2);
                d_RQ *= 2;
                d_RQ *= 2*pow(L11_i2,2)*L12_i1*Z(i1,j + dim)-2*L11_i1*L11_i2*L12_i2*(i2,j + dim)-
                        2*L11_i1*L11_i2*L12_i1*Z(i2,j + dim)+2*pow(L11_i1,2)*L12_i2*Z(i2,j + dim);
                d_RQ /= pow((pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)+pow(L22_i2,2)) + pow(L11_i2,2)*pow(L12_i1,2)-2*L11_i1*L11_i2*L12_i1*L12_i2+
                        pow(L11_i1,2)*pow(L12_i2,2),2);

                double temp = -2*L11_i1*diff(0)*diff(1)*Z(i1,j+dim)+2*pow(diff(0),2)*pow(L12_i1,2)*Z(i1,j+dim)-2*L11_i2*diff(0)*diff(1)*Z(i2,j+dim)+2*pow(diff(0),2)*pow(L12_i2,2)*Z(i2,j+dim);
                temp *= 2;
                temp /= (pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)+pow(L22_i2,2))+pow(L11_i2,2)*pow(L12_i1,2)-2*L11_i1*L11_i2*L12_i1*L12_i2+
                        pow(L11_i1,2)*pow(L12_i2,2);
                d_RQ += temp;

                dcovmat(i1,i2,j + nbasis + nisoparm) = exp(-sqrt(quad(0,0)))/sqrt(det((Sigma_i1+Sigma_i2)/2)) * 
                    (det_Sigma_i1*det_Sigma_i2*d_detSigma/(2*det((Sigma_i1+Sigma_i2)/2)) -
                    det_Sigma_i1*det_Sigma_i2*d_RQ/(2*sqrt(quad(0,0))));
                dcovmat(i2,i1,j + nbasis + nisoparm) = dcovmat(i1,i2,j + nbasis + nisoparm);
            }

            // Derivatives for covparms(2 * nbasis + nisoparm,..,3 * nbasis + nisoparm) for L22
            for (int j = 0; j < nbasis; j++) {
                double d_detSigmai14 = 0.5 * pow(det(Sigma_i1), .25) * Z(i1, j + dim);
                double d_detSigmaj14 = 0.5 * pow(det(Sigma_i2), .25) * Z(i2, j + dim);

                double d_detSigma = pow(L11_i1*L22_i1,2)*Z(i1,j + dim) + pow(L22_i1*L11_i2,2)*Z(i1,j + dim)+
                                    pow(L11_i1*L22_i2,2)*Z(i2,j + dim) + pow(L11_i2*L22_i2,2)*Z(i2,j + dim);
                d_detSigma *= 0.5;
                double d_RQ  = (pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)*pow(diff(0),2)+pow(L11_i2,2)*pow(diff(0),2)+
                               pow(L11_i1,2)*pow(diff(1),2)+pow(L11_i2,2)*pow(diff(1),2)-2*L11_i1*diff(0)*diff(1)*L12_i1+pow(diff(0),2) + 
                               pow(diff(0),2)*pow(L12_i1,2)-2*L11_i2*diff(0)*diff(1)*L12_i2+pow(diff(0),2)*pow(L12_i2,2));
                d_RQ *= 2;
                d_RQ *= 2*pow(L22_i1,2)*Z(i1,j + dim)+2*pow(L22_i2,2)*Z(i2,j + dim);
                d_RQ /= pow((pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)+pow(L22_i2,2)) + pow(L11_i2,2)*pow(L12_i1,2)-2*L11_i1*L11_i2*L12_i1*L12_i2+
                        pow(L11_i1,2)*pow(L12_i2,2),2);

                double temp = 2*pow(L22_i1,2)*pow(diff(0),2)+2*pow(L22_i2,2)*pow(diff(0),2)*Z(i2,j + dim);
                temp *= 2;
                temp /= (pow(L11_i1,2)+pow(L11_i2,2))*(pow(L22_i1,2)+pow(L22_i2,2))+pow(L11_i2,2)*pow(L12_i1,2)-2*L11_i1*L11_i2*L12_i1*L12_i2+
                        pow(L11_i1,2)*pow(L12_i2,2);
                d_RQ += temp;

                dcovmat(i1,i2,j + 2 * nbasis + nisoparm) = exp(-sqrt(quad(0,0)))/sqrt(det((Sigma_i1+Sigma_i2)/2)) * 
                    (det_Sigma_i1*d_detSigmai14 + det_Sigma_i2*d_detSigmaj14 + det_Sigma_i1*det_Sigma_i2*d_detSigma/(2*det((Sigma_i1+Sigma_i2)/2)) -
                    det_Sigma_i1*det_Sigma_i2*d_RQ/(2*sqrt(quad(0,0))));
                dcovmat(i2,i1,j + 2 * nbasis + nisoparm) = dcovmat(i1,i2,j + 2 * nbasis + nisoparm);
            }

        }
    }}

    return dcovmat;
}

#endif
