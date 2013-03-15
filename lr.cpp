//
//  lr.cpp
//  
//
//  Created by Subramanian Venkatesan on 22/02/13.
#include <iostream>
#include "matrix.h"
#include <string>
#include <limits>
#define STEP 0.01
#define EPS 1e-9
using namespace std ;
double func_val(Matrix t,Matrix x)
{
  Matrix sum = t.transpose() * x ;
  return sum[0][0];
}
Matrix gradient(Matrix t,Matrix x,Matrix y)
{      
    Matrix sum(t.getDimensions().first,1);
    for(int i = 0 ; i < y.getDimensions().first; i++)
        {	
            double yval =y[i][0];
            double f=func_val(t,x.getRowVector(i));
            sum = sum + (f - yval) * x.getRowVector(i);
               
	}
        sum = STEP * sum;
        double m = y.getDimensions().first ;
        m = 1/m ; 
        sum = m * sum ;
        return sum ; 
}
double computeCosts(Matrix t , Matrix x , Matrix y)
{
    double sum= 0 ;
    int m = y.getDimensions().first;
    for(int i = 0 ; i < m ; i++)
    {
       double h = func_val(t,x.getRowVector(i));
       double yval = y[i][0];
       sum = sum + ( h - yval ) * (h - yval);
        
   }   
    return 0.5 * sum / m ;  

}
int main()
{
  int m,n;
  cout<<"Linear Regression using Gradient Descent"<<endl;
  cout<<"No of Data points:";
  cin>>m;
  cout<<"No of Dimensions:";
  cin >> n ;
  cout<<"Enter the name of the data file:";
  string filename;
  cin>>filename;
  Matrix A(m,n,filename);
  vector<double> x ; 
  vector<double> y = A.getColumnVector(1); 
  Matrix X(m,n) ;

 // adding ones X[0] vector 
  for(int i= 0 ; i< m ; i++)
  {
      X[i][0] = 1;
  }
  // Adding the remaining column vectors of x
  for(int i= 0;i < n - 1 ;i++)
  {
      
      vector<double> x = A.getColumnVector(i);
      for(int j = 0 ; j < x.size() ; j = j + 1)
     {
      
        X[j][i+1] = x[j] ; 
     } 
    
  }  

  Matrix Y(y.size(),1); // output vector
  for(int i = 0 ; i < y.size() ; i++ )
  {
      Y[i][0] = y[i] ;	
  }
 
  Matrix w(X.getDimensions().second,1) ;  // Initial vector which is set to '0'
  Matrix g = gradient(w,X,Y); // intial gradient
  while( g.magnitude() > EPS   )   // magnitude return the norm of the gradient 
  {
      w = w  - g  ; // weight update    
      g = gradient(w,X,Y);    
      cout<<computeCosts(w,X,Y)<<endl; // the cost has to be decreasing
  } 
  cout<<"Weight Vector";
  cout << w.transpose(); // final w * 
  
}
