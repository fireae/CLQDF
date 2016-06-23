
#ifndef	CMATRIX_H
#define	CMATRIX_H

class CMatrix
{
	// Diagonalization (K-L transform)
	void Jacobi( float**, int );
	void planeRotate( float**, int, int, int, float** );

	void Householder( float**, int, float** );
	int reflection( float*, int, int, float* );
	void updateMat( float**, float*, int, int );

	int QLdecomp( float**, int, int, float** );
	float shiftQL( float**, int, int, float** );

public:
	float** allocMat( int );
	void freeMat( float**, int );
	void identiMat( float**, int );	// set identity matrix

	// Elementary matrix and vector compuations
	void outProduct( float*, float*, int, float** );		// outer product
	float innProduct( float*, float*, int );		// inner product

	void matXvec( float**, float*, int, float* );	// multiplication
	void matXmat( float**, float**, int, float** );
	void matXmat( float**, float**, int );
	void matCopy( float**, int, float** );

	void transpose( float**, int, float** );

	void diagonalize( float**, int, float** );

	// Check the orthonormality and characteristic equation
	float characteristic( float**, int, float*, float*, int );
	float KLT( float**, int, float*, float* );
	float PCA( float**, int, float*, float*, int );

	float matInverse( float**, int, float** );
};

#endif

