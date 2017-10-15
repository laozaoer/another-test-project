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
//'require(CDLasso)
//'X=matrix(c(1,1,-1,-1),nrow=2)
//'y=c(2,-2)
//'lambda=1
//'c=0.01
//'CDLasso(y,X,lambda,c)
// [[Rcpp::export]]
arma::vec CDLasso(const arma::vec& y,const arma::mat& X,double lambda,double c){
  arma::vec betahat;
  arma::vec r;
  int p=X.n_cols;
  betahat.zeros(p);
  r=y-X*betahat;
  int n=X.n_rows;
  arma::vec tempbeta;
  do
  {
    tempbeta=betahat;
    for(int i=0;i<p;i++)
    {
      double z=sum(r%X.col(i))/n+betahat(i);
      double betaplus;
      if(z>lambda)
      {
        betaplus=z-lambda;
      }
      else if(z<(-lambda))
      {
        betaplus=z+lambda;
      }
      else
      {
        betaplus=0;
      }
      r=r+X.col(i)*(-betaplus+betahat(i));
      betahat.at(i)=betaplus;
    }
  }while(sum((tempbeta-betahat)%(tempbeta-betahat))>c);
  return betahat;
}
