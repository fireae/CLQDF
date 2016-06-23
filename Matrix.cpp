#include <string.h>
#include <math.h>
#include "Matrix.h"

#define TZERO 1E-6

// Inersion of symmetrix matrix
// return the logarithm of determinant
float CMatrix::matInverse( float** C, int n, float** B )
{
	float **A, ** P;
	A = allocMat( n );
	P = allocMat( n );

	int i;
	for( i=0; i<n; i++ )
		memcpy( A[i], C[i], n*sizeof(float) );	// A: copy of covariance matrix

	diagonalize( A, n, P );

	float ldet = 0;
	for( i=0; i<n; i++ )
	{
		if( A[i][i]<TZERO )
			A[i][i] = (float)TZERO;
		ldet += (float)log( A[i][i] );
	}

	// A is now a diagonal matrix
	for( i=0; i<n; i++ )
	{
		if( A[i][i]>0 )
			A[i][i] = 1/A[i][i];	// inverse of A
	}

	matXmat( P, A, n, B );

	transpose( P, n, A );
	matXmat( B, A, n );		// results saved in B

	/*matXmat( C, B, n, A );	// verify the inverse matrix

	int j;
	float sum = 0;
	for( i=0; i<n; i++ )
		for( j=0; j<n; j++ )
		{
			if( i==j )
				sum += fabs( 1-A[i][j] );
			else
				sum += fabs( A[i][j] );
		}*/

	freeMat( P, n );
	freeMat( A, n );

	return ldet;
}

// Check the orthonormality and characteristic equation
float CMatrix::characteristic( float** C, int dim, 
							   float* phi, float* lambta, int vecNum )
{
	float* tvec;
	tvec = new float [dim];

	int i, j;
	float ip, aver;	// inner product
	aver = 0;
	for( i=0; i<vecNum; i++ )
	{
		ip = innProduct( phi+i*dim, phi+i*dim, dim );	// normality
		aver += (float)fabs( ip-1 );
		for( j=0; j<i; j++ )
		{
			ip = innProduct( phi+i*dim, phi+j*dim, dim );	// orthogonality
			aver += (float)fabs( ip );
		}
	}
	aver /= vecNum*(vecNum+1)/2;

	float sum, total;

	total = 0;
	for( j=0; j<vecNum; j++ )
	{
		if( lambta[j]<0.0001 )
			continue;
		matXvec( C, phi+j*dim, dim, tvec );
		for( i=0; i<dim; i++ )
			tvec[i] -= lambta[j]*phi[j*dim+i];
		sum = innProduct( tvec, tvec, dim )/(lambta[j]*lambta[j]);
		total += sum;
	}
	total /= vecNum;

	delete []tvec; tvec=NULL;

	return aver+total;
}

// Compute the principal eigenvectors and eigenvalues
float CMatrix::PCA( float** C, int n, float* phi, float* lambta, int vecNum )
{
	float **A, **P;

	A = allocMat( n );
	P = allocMat( n );	// P stores the orthognal transformation matrix

	int i;
	for( i=0; i<n; i++ )
		memcpy( A[i], C[i], n*sizeof(float) );	// A: copy of covariance matrix

	diagonalize( A, n, P );

	// Sort the eigen values in decreasing order
	char* mark;
	mark = new char [n];
	memset( mark, 0, n );

	float total = 0;	// sum of eigenvalues
	int j, k;
	for( int m=0; m<vecNum; m++ )
	{
		lambta[m] = -1;
		for( k=n-1; k>=0; k-- )		// The large eigenvalues usually at last
		{
			if( mark[k] ||k<0)
				continue;

			if( A[k][k]>lambta[m] )
			{
				lambta[m] = A[k][k];
				j = k;		// j: the position of largest eigenvalue
			}
		}
		mark[j] = 1;

		total += lambta[m];
		for( i=0; i<n; i++ )
			phi[m*n+i] = P[i][j];	// column of transform matrix as eigen vector
	}

	freeMat( A, n );
	freeMat( P, n );
	delete []mark; mark=NULL;

	return total;
}

// Compute the eigenvectors and eigenvalues of a symmetric matrix
float CMatrix::KLT( float** C, int n, float* phi, float* lambta )
{
	float **A, **P;

	A = allocMat( n );
	P = allocMat( n );	// P stores the orthognal transformation matrix

	int i;
	for( i=0; i<n; i++ )
		memcpy( A[i], C[i], n*sizeof(float) );	// A: copy of covariance matrix

	diagonalize( A, n, P );

	// Sort the eigen values in decreasing order
	char* mark;
	mark = new char [n];
	memset( mark, 0, n );

	float total = 0;	// sum of eigenvalues, equals the variance
	int j, k;
	for( int m=0; m<n; m++ )
	{
		lambta[m] = -1;
		for( k=n-1; k>=0; k-- )		// The large eigenvalues usually at last
		{
			if( mark[k] )
				continue;

			if( A[k][k]>lambta[m] )
			{
				lambta[m] = A[k][k];
				j = k;		// j: the position of largest eigenvalue
			}
		}
		mark[j] = 1;

		total += lambta[m];
		for( i=0; i<n; i++ )
			phi[m*n+i] = P[i][j];	// column of transform matrix as eigen vector
	}

	freeMat( A, n );
	freeMat( P, n );
	delete []mark; mark=NULL;

	return total;
}

// Diagonalization of symmetric matrix
// P: transformation matrix
void CMatrix::diagonalize( float** A, int n, float** P )
{
	identiMat( P, n );	// initially identity matrix
	Householder( A, n, P );	// tri-diagonalization

	float* d;		// diagonal elements (eigen values)
	d = new float [n];

	int k;
	for( k=0; k<n-1; k++ )
	{
		d[k] = shiftQL( A, n, k, P );
		if( k>0 )
			d[k] += d[k-1];
	}
	d[n-1] = d[n-2]+A[n-1][n-1];

	for( k=0; k<n; k++ )
		A[k][k] = d[k];

	delete []d; d=NULL;
}

// QL method with acceleration shifts
float CMatrix::shiftQL( float** A, int n, int k, float** P )
{
	float eigv = 0;
	float s1, s2;
	float b, c, sq, t;
	int i;

	while( fabs( A[k][k+1] )>TZERO )
	{
		b = -(A[k][k]+A[k+1][k+1]);
		c = A[k][k]*A[k+1][k+1]-A[k][k+1]*A[k][k+1];
		t = b*b-4*c;
		if( t<0 )		// modified 2006.06.08, found by Tianfu Gao
			sq = 0;
		else
			sq = (float)sqrt( t );
		s1 = (-b+sq)/2;
		s2 = (-b-sq)/2;
		if( fabs( A[k][k]-s1 )>fabs( A[k][k]-s2 ) )		// s2 is closer to d1
			s1 = s2;

		eigv += s1;

		for( i=k; i<n; i++ )
			A[i][i] -= s1;
		QLdecomp( A, n, k, P );
	}
	if( fabs( A[k][k] )>TZERO )
	{
		s1 = A[k][k];
		eigv += s1;
		for( i=k; i<n; i++ )
			A[i][i] -= s1;
	}

	return eigv;
}

// Decompose a tridiagonal matrix into an orthogonal matrix and a lower triangular one
// Process the sub-matrix starting with a specified row
int CMatrix::QLdecomp( float** A, int n, int row, float** P )
{
	float *d, *e;
	float *p, *q;
	float *c1, *c2;

	d = new float [n];
	e = new float [n];
	p = new float [n];
	q = new float [n];
	c1 = new float [n];
	c2 = new float [n];

	int i, j;
	for( i=0; i<n; i++ )
		d[i] = A[i][i];
	for( i=0; i<n-1; i++ )
		e[i] = A[i+1][i];

	p[n-1] = d[n-1];
	q[n-2] = e[n-2];

	float** Q;
	Q = allocMat( n );

	identiMat( Q, n );	// initially be an identity matrix

	int flag = 1;	// remains 1 if all off-diagonal elements are 0

	float c, s, t;
	for( int k=n-2; k>=row; k-- )
	{
		if( fabs( e[k] )<TZERO )		// c=1, s=0
		{
			p[k] = d[k];
			if( k>row )
				q[k-1] = e[k-1];

			continue;
		}

		flag = 0;	// at least one off-diagonal element is not 0

		t = -p[k+1]/e[k];
		s = 1/(float)sqrt(1+t*t);
		c = s*t;

		p[k] = c*d[k]+s*q[k];
		if( k>row )
			q[k-1] = c*e[k-1];
		p[k+1] = -s*e[k]+c*p[k+1];
		q[k] = -s*d[k]+c*q[k];

		// update the transform matrix Q
		for( i=row; i<n; i++ )
		{
			c1[i] = c*Q[i][k]+s*Q[i][k+1];	// column k
			c2[i] = -s*Q[i][k]+c*Q[i][k+1];	// column k+1
		}
		for( i=row; i<n; i++ )
		{
			Q[i][k] = c1[i];
			Q[i][k+1] = c2[i];
		}

		// update the transform matrix P
		for( i=0; i<n; i++ )
		{
			c1[i] = c*P[i][k]+s*P[i][k+1];	// column k
			c2[i] = -s*P[i][k]+c*P[i][k+1];	// column k+1
		}
		for( i=0; i<n; i++ )
		{
			P[i][k] = c1[i];
			P[i][k+1] = c2[i];
		}
	}

	// Calculate the transformed matrix A=LQ (symmetric tridiagonal)
	for( i=row; i<n; i++ )
		for( j=row; j<n; j++ )
		{
			if( i-j>1 || j-i>1 )
				A[i][j] = 0;
			else if( i==0 )
				A[i][j] = p[i]*Q[i][j];
			else if( j>=i )
				A[i][j] = q[i-1]*Q[i-1][j]+p[i]*Q[i][j];
			else
				A[i][j] = A[j][i];
		}

	freeMat( Q, n );

	delete []d; d=NULL;
	delete []e; e=NULL;
	delete []p; p=NULL;
	delete []q; q=NULL;
	delete []c1; c1=NULL;
	delete []c2; c2=NULL;

	return flag;
}

// Tridiagonalization of symmetric matrix by Householder's method
// the transformation matrix is stored in P
void CMatrix::Householder( float** A, int n, float** P )
{
	float *W, *V, *Q;
	float coef;

	W = new float [n];
	V = new float [n];
	Q = new float [n];

	int k, i, j;
	for( k=0; k<n-2; k++ )
	{
		if( reflection( A[k], n, k, W ) )
			continue;

		updateMat( P, W, n, k );	// update the transformation matrix P
		
		matXvec( A, W, n, V );
		coef = innProduct( W, V, n );
		for( i=0; i<n; i++ )
			Q[i] = V[i]-coef*W[i];

		for( i=0; i<n; i++ )
			for( j=0; j<n; j++ )
				A[i][j] = A[i][j]-2*W[i]*Q[j]-2*Q[i]*W[j];
	}

	delete []W; W=NULL;
	delete []V; V=NULL;
	delete []Q; Q=NULL;
}

// update the transformation matrix in Housholder
void CMatrix::updateMat( float** P, float* W, int n, int k )
{
	if( k==0 )
	{
		int i, j;
		for( i=1; i<n; i++ )
			for( j=1; j<n; j++ )
				P[i][j] = P[i][j]-2*W[i]*W[j];

		return;
	}

	float* tv;
	tv = new float [n];

	int i, j;
	for( i=0; i<n; i++ )
	{
		tv[i] = 0;
		for( j=k+1; j<n; j++ )
			tv[i] += P[i][j]*W[j];
	}

	for( i=0; i<n; i++ )
		for( j=0; j<n; j++ )
			P[i][j] = P[i][j]-2*tv[i]*W[j];

	delete []tv; tv=NULL;
}

// Householder reflection of a vector
int CMatrix::reflection( float* X, int n, int k, float* W )
{
	float S2 = 0;
	int i;
	for( i=k+1; i<n; i++ )
		S2 += X[i]*X[i];

	if( S2==0 )
		return 1;

	float S;
	S = (float)sqrt( S2 );
	if( X[k+1]<0 )
		S = -S;

	float R2, R;
	R2 = 2*X[k+1]*S+2*S2;
	R = (float)sqrt( R2 );

	for( i=0; i<n; i++ )
	{
		if( i<=k )
			W[i] = 0;
		else if( i==k+1 )
			W[i] = (X[k+1]+S)/R;
		else
			W[i] = X[i]/R;
	}

	return 0;
}

// Symmetric matrix diagonalization by Jacobi's method
// very slow for large dimension
void CMatrix::Jacobi( float** A, int n )
{
	float** D;
	D = allocMat( n );

	int i, j;
	int mi, mj;
	float av, maxv;

	for( i=0; i<n; i++ )
		memcpy( D[i], A[i], n*sizeof(float) );

	// Iterative plane rotation to TZERO-out the maximum off-diagonal element
	for( int cycle=0; cycle<2*n*n; cycle++ )
	{
		maxv = 0;
		for( i=0; i<n; i++ )
			for( j=i+1; j<n; j++ )
			{
				av = (float)fabs( A[i][j] );
				if( av>maxv )
				{
					maxv = av;
					mi = i;
					mj = j;
				}
			}

		if( maxv<0.0001 )
			break;

		planeRotate( A, n, mi, mj, D );
		for( i=0; i<n; i++ )
			memcpy( A[i], D[i], n*sizeof(float) );
	}

	freeMat( D, n );
}

// Rotation of a plane composed of two axes to TZERO-out a[p][q]
// the rotated matrix is stored in D[][]
void CMatrix::planeRotate( float** A, int n, int p, int q, float** D )
{
	float theta, t, c, s;
	theta = (float)0.5*(A[q][q]-A[p][p])/A[p][q];
	if( theta>0 )
		t = -theta+(float)sqrt(theta*theta+1);
	else
		t = -theta-(float)sqrt(theta*theta+1);
	c = 1/(float)sqrt(t*t+1);	// cos
	s = c*t;			// sin

	D[p][q] = 0;
	D[q][p] = 0;
	D[p][p] = c*c*A[p][p]+s*s*A[q][q]-2*c*s*A[p][q];
	D[q][q] = s*s*A[p][p]+c*c*A[q][q]+2*c*s*A[p][q];

	for( int j=0; j<n; j++ )
	{
		if( j==p || j==q )
			continue;

		D[j][p] = c*A[j][p]-s*A[j][q];
		D[p][j] = D[j][p];
		D[j][q] = s*A[j][p]+c*A[j][q];
		D[q][j] = D[j][q];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Elementary matrix and vector operations

float** CMatrix::allocMat( int n )
{
	float** A;
	A = new float* [n];

	for( int i=0; i<n; i++ )
		A[i] = new float [n];

	return A;
}

void CMatrix::freeMat( float** A, int n )
{
	for( int i=0; i<n; i++ )
	{
		delete []A[i]; A[i]=NULL;
	}
	delete []A; A=NULL;
}

// Set a matrix to be identity matrix
void CMatrix::identiMat( float** A, int n )
{
	int i, j;
	for( i=0; i<n; i++ )
		for( j=0; j<n; j++ )
		{
			if( i==j )
				A[i][j] = 1;
			else
				A[i][j] = 0;
		}
}

void CMatrix::matCopy( float** A, int n, float** B )
{
	for( int i=0; i<n; i++ )
		memcpy( B[i], A[i], n*sizeof(float) );
}

void CMatrix::transpose( float** A, int n, float** B )
{
	int i, j;
	for( i=0; i<n; i++ )
		for( j=0; j<n; j++ )
			B[i][j] = A[j][i];
}

// matrix-matrix multiplication, result stored in a 3rd matrix
void CMatrix::matXmat( float** A, float** B, int n, float** C )
{
	int i, j, k;
	for( i=0; i<n; i++ )
		for( j=0; j<n; j++ )
		{
			C[i][j] = 0;
			for( k=0; k<n; k++ )
				C[i][j] += A[i][k]*B[k][j];
		}
}

// matrix-matrix multiplication, result overrides the 1st matrix
void CMatrix::matXmat( float** A, float** B, int n )
{
	float* trow;
	trow = new float [n];

	int i, j, k;
	for( i=0; i<n; i++ )
	{
		for( j=0; j<n; j++ )
		{
			trow[j] = 0;
			for( k=0; k<n; k++ )
				trow[j] += A[i][k]*B[k][j];
		}
		memcpy( A[i], trow, n*sizeof(float) );
	}

	delete []trow; trow=NULL;
}

// matrix-vector multiplication
void CMatrix::matXvec( float** A, float* vect, int n, float* outv )
{
	int i, j;
	for( i=0; i<n; i++ )
	{
		outv[i] = 0;
		for( j=0; j<n; j++ )
			outv[i] += A[i][j]*vect[j];
	}
}

// Outer product of two vectors
void CMatrix::outProduct( float* v1, float* v2, int n, float** A )
{
	int i, j;
	for( i=0; i<n; i++ )
		for( j=0; j<n; j++ )
			A[i][j] = v1[i]*v2[j];
}

// Inner product of two vectors
float CMatrix::innProduct( float* v1, float* v2, int n )
{
	float inp = 0;
	for( int i=0; i<n; i++ )
		inp += v1[i]*v2[i];

	return inp;
}

