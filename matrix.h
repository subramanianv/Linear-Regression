#include<iostream>
#include<vector>
#include<cassert>
#include<stdio.h>
#include<math.h>
using namespace std;
class Matrix
{
    private:
        vector<vector<double> > mat;
        int row,column;
    public:           
        friend ostream& operator<<(ostream& output, const Matrix& p);
        friend Matrix operator + (Matrix a, Matrix b);
        friend Matrix operator * (double k , Matrix b);
        friend Matrix operator * (int k, Matrix b);
        friend Matrix operator * (Matrix a, Matrix b);
        friend Matrix operator - (Matrix a, Matrix b);
        friend Matrix operator - (Matrix a);
        Matrix(int rows,int cols)
        {
          assert(rows > 0 );
          assert(cols > 0);
          row=rows;
          column=cols;
          for(int i=0;i<rows;i++)
          {
               vector<double> v(cols);
               for(int j=0;j<cols;j++)
               {
               	v[j]=0.0;
               }
               mat.push_back(v);
          }    
        }
        //Constructor #2 //
        Matrix(int rows,int cols, const string filename)
        {
        
            FILE* fp;
            char * cstr = new char [filename.length()+1];
            std::strcpy (cstr, filename.c_str());
            fp=fopen(cstr,"r");
            assert(fp!=NULL);
            assert(rows > 0 );
            assert(cols > 0);
            row=rows;
            column=cols;
            for (int i=0; i<rows; i++) 
            {
                vector<double> v(cols);
                for(int j=0;j<cols;j++)
                {
                    double x;
                    fscanf(fp,"%lf",&x);
                    v[j]=x;
                        
                 }
                    mat.push_back(v);    
            }
        
        
        }
        
        pair<int,int> getDimensions()
        {
            return make_pair(row,column);
        }
        vector<double>& operator[](int i)
        {
            assert(i>=0 && i<row);
            return mat[i];
        }
        
        Matrix transpose()
        {
            Matrix mt=Matrix(column,row);
            for (int i=0; i<row; i++)
            {
                for (int j=0; j<column; j++)
                {
                    mt[j][i]=mat[i][j];
                }
            
             }
             return mt;
        }
       // LU Decomposition //
        void decompose(Matrix &L , Matrix &U)
        {
        
            // Adopted from CRLS book.
    	    L=Matrix::identity(row);
            vector<vector<double> > mat_copy= mat;
            int n=row;
            for (int k=0; k<n; k++)
            {
                U[k][k]=mat[k][k];
                for (int i=k+1; i<n; i++)
                {
                    if (U[k][k]==0) 
	                L[i][k]=mat[i][k] / 1;
                    else
                        L[i][k]=mat[i][k] / U[k][k];
            
                    U[k][i] = mat[k][i];
                
            
                }
                for (int i=k+1; i<n;i++) 
                {
                     for (int j=k+1; j < n  ;j++) 
		         mat[i][j] = mat[i][j] - L[i][k] * U[k][j];
            
        	
            
       	        }
            
            }
        
        
        mat = mat_copy;
    
    }
    static Matrix identity(int r)
    {
        Matrix A(r,r);

        for (int i=0; i<r; i++)
        {
            
            A[i][i]=1.0;
        }
            return A;
    }
    
    
    double determinant()
    {
        assert(row == column );
        Matrix L= Matrix :: identity(row);
        Matrix U(row,row);
        decompose(L,U);
        double val=1.0;
        for (int i=0; i<row; i++) 
            val  =val *  L[i][i] * U[i][i];
    
    
         return val;
    }
   double magnitude()
    {
        assert(column == 1);
        double sum =0;
        for(int i=0;i<row; i++)
             sum = sum + mat[i][0] * mat[i][0];
        return sqrt(sum);
    }

    vector<double> getColumnVector(int c)
    {
         assert(c<column);
         vector<double> col;
         for(int i=0;i<row;i++)
         {
	    col.push_back(mat[i][c]);
	  }
          return col;
    }
   Matrix getRowVector(int r)
   {
       assert(r < row);
       Matrix m(column,1);
       for(int i=0;i<column;i++)
       {
	 m[i][0] = mat[r][i] ;
         
	}
        return m ;
   }
        
};
// Friend Functions //


Matrix operator+(Matrix a,Matrix b)
{
    assert(a.row == b.row && a.column == b.column );
        Matrix c(a.row,a.column);
        for (int i=0; i<a.row; i++)
        {
            for (int j=0; j< a.column; j++)
            {
                c[i][j] = a[i][j] + b[i][j]; 
            }
        }
    return c;
}

Matrix operator-(Matrix a,Matrix b)
{
    assert(a.row == b.row && a.column == b.column );
    Matrix c(a.row,a.column);
    for (int i=0; i<a.row; i++)
        {
        for (int j=0; j< a.column; j++)
            {
            c[i][j] = a[i][j] - b[i][j]; 
            }
        }
    return c;
}

ostream& operator<<(ostream& output, const Matrix& p)
{
     output<<endl;
     for(int i=0;i<p.row;i++)
     {
        for(int j=0;j<p.column;j++)
           output<<p.mat[i][j]<<" ";
        output<<endl;
     }
        
}
Matrix operator * (int k, Matrix b)
{
    Matrix A(b.row,b.column);
    for(int i=0;i<b.row;i++)
    {
    for(int j=0; j < b.column ; j++)
        A[i][j] = k* b[i][j];
    }
    return A;
}
Matrix operator * (double k, Matrix b)
{
    Matrix A(b.row,b.column);
    for(int i=0;i<b.row;i++)
    {
    for(int j=0; j < b.column ; j++)
        A[i][j] = k* b[i][j];
    }
    return A;
}

Matrix operator * (Matrix a, Matrix b)
{
    assert(a.column == b.row);
    int cRow=a.row,cCol=b.column;
    Matrix c(cRow,cCol);
    for (int i=0; i<cRow; i++) {
        for (int j=0; j<cCol; j++) {
            for (int k=0; k<b.row; k++) {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
            }
        }
    }
    
    return c;
}
Matrix operator - (Matrix a)
{
    Matrix c(a.row,a.column);
    for (int i=0; i<a.row; i++) {
        for (int j=0; j<a.column; j++) {
            
            c[i][j]=-a[i][j];
        }
    }
    
    return c;
}

