#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]
//'Two method for Lasso
//'
//'@param y the response vector
//'@param X the design matrix
//'@param lambda the penalty parameter
//'@param c the precision parameter
//'@return the estimated beta
//'@example
//'require(POLasso)
//'X=matrix(c(1,1,-1,-1),nrow=2)
//'y=c(2,-2)
//'lambda=1
//'c=0.01
//'POLasso(y,X,lambda,c)
// [[Rcpp::export]]
arma::vec POLasso(arma::vec& y,arma::mat& X,double lambda,double c){
  arma::vec betahat;
  arma::vec r;
  int p=X.n_cols;
  betahat.zeros(p);
  r=y-X*betahat;
  int n=X.n_rows;
  double M;
  arma::vec eigenvalue=eig_sym(X.t()*X);
  M=max(eigenvalue)/n;
  arma::vec tempbeta;
  do
  {
    tempbeta=betahat;
    arma::vec SFbeta=betahat+(X.t()*(y-X*betahat))/(n*M);
    for(int i=0;i<p;i++)
    {
      if(SFbeta(i)>(lambda/M))
      {
        betahat(i)=SFbeta(i)-(lambda/M);
      }
      else if(SFbeta(i)<(-(lambda/M)))
      {
        betahat(i)=SFbeta(i)+(lambda/M);
      }
      else
      {
        betahat(i)=0;
      }
    }
  } while (sum((tempbeta-betahat)%(tempbeta-betahat))>c);
  return betahat;
}