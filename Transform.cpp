#include <stdio.h>
#include <string.h>
#include <math.h>
#include "MQDF_training.h"

extern float typeMulti;		// defined in trainMain.cpp

// Power transformation of one feature vector
void powerTrans( unsigned char* vect, int dim, float power, float* fvect )
{
	for( int i=0; i<dim; i++ )
	{
		if( power==1 )
			fvect[i] = (float)vect[i];
		else if( power==0.5 )
		{			if( vect[i]<0 )
				fvect[i] = -sqrt( -(float)vect[i] );
			else
				fvect[i] = sqrt( (float)vect[i] );
		}
		else
		{
			if( vect[i]<0 )
				fvect[i] = -pow( -(float)vect[i], power );
			else
				fvect[i] = pow( (float)vect[i], power );
		}
	}
}

float innerProd( float* vec1, float* vec2, int dim )
{
	float sum = 0;
	for( int i=0; i<dim; i++ )
		sum += vec1[i]*vec2[i];

	return sum;
}

// Transform feature valut to (-1,+1)
float bipolar( float a )
{
	float exv2;
	exv2 = (float)exp( -2*a );
	return (1-exv2)/(1+exv2);
}

// Gram-Schmidt orthonormalization, common for multiple classes
// 1: normalizaiton only; 2: orthonromalization
void GramSchmidt( float* vects, int dim, int vnum, int ortho )
{
	float proj, coef;
	int i, j, k;

	for( j=0; j<vnum; j++ )
	{
		if( ortho==2 )
		{
			for( k=0; k<j; k++ )
			{
				proj = innerProd( vects+k*dim, vects+j*dim, dim );
				for( i=0; i<dim; i++ )
					vects[j*dim+i] -= proj*vects[k*dim+i];
			}
		}

		proj = innerProd( vects+j*dim, vects+j*dim, dim );
		coef = (float)sqrt( proj );
		for( i=0; i<dim; i++ )
			vects[j*dim+i] /= coef;
	}
}

// Offset the type-dependent multiplier
void makeInput( float* ftr, int dim, float* input )
{
	for( int i=0; i<dim; i++ )
		input[i] = (float)ftr[i]/typeMulti;
}

///////////////////////////////////////////////////////////////////////////////////

CTransform::CTransform()
{
}

CTransform::~CTransform()
{
}

// Data rescaling with power transformation and subtraction
void CTransform::rescaling( unsigned char* data, int sampNum, int dim, float power, float* gmean,int bipol, float scale1, float* trData )
{
	float* temp;		// intermediate vector of power transforamtion
	temp = new float [dim];

	int n, i;
	for( n=0; n<sampNum; n++ )
	{
		powerTrans( data+(int64_t )n*dim, dim, power, temp );
		for( i=0; i<dim; i++ )
			trData[(int64_t )n*dim+i] = transType( scale1*(temp[i]-gmean[i]), bipol );
	}

	delete temp;
}

float CTransform::transType( float va, int bipol )
{
	float value;
	if( bipol )
		value = (float)1.313*bipolar( va );
	else
		value = va;

	value *= typeMulti;		// multiplier depending on transformed data type

	if( sizeof(float)==4 )		// float
		return (float)(value);

	value += 0.5;
	if( sizeof(float)==2 )	// short
	{
		if( value>10000 )
			value = 10000;
		else if (value<-10000 )
			value = -10000;
	}
	else	// char
	{
		if( value>127 )
			value = 127;
		else if( value<-127 )
			value = -127;
	}
	return (float)value;
}

// Feature transformation to PCA, whitening or Fisher subspace
void CTransform::reduceTrans( unsigned char* data, int sampNum, int dim, float power, float* mean, float* basis, int redDim, int residual, int bipol, float scale1, float* trData )
{
	float *temp;		// intermediate vector of power transformation
	temp = new float [dim];

	int enDim;
	if( residual )	// residual of subspace projection
		enDim = redDim+1;
	else
		enDim = redDim;

	int n, i, j;
	float proj, euclid;
	for( n=0; n<sampNum; n++ )
	{
		powerTrans( data+(int64_t )n*dim, dim, power, temp );
		for( i=0; i<dim; i++ )
			temp[i] -= mean[i];		// shift w.r.t gross mean
		if( residual )
		{
			euclid = 0;
			for( i=0; i<dim; i++ )
				euclid += temp[i]*temp[i];		// square Euclidean distance
		}

		for( j=0; j<redDim; j++ )
		{
			proj = innerProd( temp, basis+j*dim, dim );
			if( residual )
				euclid -= proj*proj;

			trData[(int64_t )n*enDim+j] = transType( proj*scale1, bipol );
		}
		if( residual )
			trData[(int64_t )n*enDim+redDim] = transType( euclid*scale1*scale1, bipol );
	}

	delete temp;
}

// Scaling factor estimation, from positive samples only
// Power transformation is integrated so as to save intermediate memory
float CTransform::dataScale( unsigned char* data, int sampNum, int dim, short* truth, float power, float* mean )
{
	int posiNum;		// number of positive samples
	posiNum = grossMean( data, sampNum, dim, truth, power, mean );

	float* temp;
	temp = new float [dim];

	float* sqnorm;		// dimension-wise square norm
	sqnorm = new float [dim];
	memset( sqnorm, 0, dim*sizeof(float) );

	int n, i;
	for( n=0; n<sampNum; n++ )
	{
		if( truth[n]<0 )
			continue;

		powerTrans( data+(int64_t )n*dim, dim, power, temp );
		for( i=0; i<dim; i++ )
		{
			temp[i] -= mean[i];		// shift w.r.t gross mean
			sqnorm[i] += temp[i]*temp[i];
		}
	}
	for( i=0; i<dim; i++ )
		sqnorm[i] /= posiNum;		// average over samples

	float sqmax = 0;		// maximum of square norm
	for( i=0; i<dim; i++ )
	{
		if( sqnorm[i]>sqmax )
			sqmax = sqnorm[i];
	}

	delete sqnorm;
	delete temp;

	return 1/(float)sqrt(sqmax);
}

// Learning the weights of whitening subspace
// Dimensionality reduction is performed by PCA
float CTransform::whitenLearn( unsigned char* data, int sampNum, int dim, short* truth, int classNum, float power, float* gmean, float* basis, int redDim )
{
	int* csnum;		// classwise number of samples
	csnum = new int [classNum];

	float* cmeans;	// class mean vectors
	cmeans = new float [classNum*dim];

	float** Cwit;	// winthin-class common covariance
	Cwit = theMat.allocMat( dim );

	commCova( data, sampNum, dim, truth, classNum, power, cmeans, Cwit, csnum );

	float *phi;		// eigenvectors of PCA subspace
	int i, j;
	if( redDim<dim )	// dimensionality reduction by PCA
	{
		phi = new float [redDim*dim];	// columns of PCA orthogonal matrix

		PCALearn( data, sampNum, dim, truth, power, gmean, phi, redDim );

		float* temp;		// columns of an intermediate matrix
		temp = new float [dim*redDim];
		for( j=0; j<redDim; j++ )
			theMat.matXvec( Cwit, phi+j*dim, dim, temp+j*dim );

		// reduced within-class covariance matrix
		for( i=0; i<redDim; i++ )
			for( j=0; j<redDim; j++ )
				Cwit[i][j] = innerProd( phi+i*dim, temp+j*dim, dim );

		delete []temp; temp=NULL;	// *phi is used yet
	}
	else
	{
		int posiNum = 0;
		int ci;
		for( ci=0; ci<classNum; ci++ )
			posiNum += csnum[ci];

		for( i=0; i<dim; i++ )
		{
			gmean[i] = 0;
			for( ci=0; ci<classNum; ci++ )
				gmean[i] += csnum[ci]*cmeans[ci*dim+i];
			gmean[i] /= posiNum;		// gross mean
		}
	}

	// within-class covariance: redDim*redDim
	float** P;
	P = theMat.allocMat( redDim );
	theMat.diagonalize( Cwit, redDim, P );		// P: orthogonal matrix
	for( i=0; i<redDim; i++ )
		Cwit[i][i] = (float)sqrt( 1./Cwit[i][i] );	// diagonal

	theMat.matXmat( P, Cwit, redDim );		// multiplication saved in P
	theMat.transpose( P, redDim, Cwit );	// transpose saved in Cwit for whitening

	if( redDim<dim )	// combine whitening and PCA eigenvectors
	{
		int k;
		for( i=0; i<redDim; i++ )
			for( j=0; j<dim; j++ )
			{
				basis[i*dim+j] = 0;
				for( k=0; k<redDim; k++ )
					basis[i*dim+j] += Cwit[i][k]*phi[k*dim+j];
			}

		delete []phi; phi=NULL;
	}
	else	// redDim==dim, basis vectors stored in Cwit
	{
		for( i=0; i<dim; i++ )
			memcpy( basis+i*dim, Cwit[i], dim*sizeof(float) );
	}

	theMat.freeMat( P, redDim );
	theMat.freeMat( Cwit, dim );

	delete cmeans;
	delete csnum;

	return 0.5;		// scale 0.5 at default
}

////////////////////////////////////////////////////////////////////////////////////

// Fisher linear discriminant analysis
float CTransform::FisherLearn( unsigned char* data, int sampNum, int dim, short* truth, int classNum,float power, float* gmean, float* weight, int redDim, float beta )
{
	int* csnum;		// classwise number of samples
	csnum = new int [classNum];

	float* cmeans;	// class mean vectors
	cmeans = new float [classNum*dim];

	float** Cwit;	// winthin-class covariance
	Cwit = theMat.allocMat( dim );

	commCova( data, sampNum, dim, truth, classNum, power, cmeans, Cwit, csnum );

	int posiNum = 0;		// number of positive samples
	int ci;
	for( ci=0; ci<classNum; ci++ )
		posiNum += csnum[ci];

	int i, j;
	for( i=0; i<dim; i++ )
	{
		gmean[i] = 0;
		for( ci=0; ci<classNum; ci++ )
			gmean[i] += csnum[ci]*cmeans[ci*dim+i];
		gmean[i] /= posiNum;		// gross mean
	}

	// Between-class covariance on whitened means
	float** Cbet;
	Cbet = theMat.allocMat( dim );

	float *temp;
	temp = new float [dim];

	if( beta>0 )	// interpolate between FDA and PCA
	{
		// Cbet must be computed on un-whitened class means
		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				Cbet[i][j] = 0;

		for( ci=0; ci<classNum; ci++ )
		{
			for( i=0; i<dim; i++ )
				temp[i] = cmeans[ci*dim+i]-gmean[i];	// shift with gross mean
			for( i=0; i<dim; i++ )
				for( j=i; j<dim; j++ )
					Cbet[i][j] += csnum[ci]*temp[i]*temp[j];
		}
		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				Cbet[i][j] /= posiNum-classNum;		// unbiased estimate
		for( i=1; i<dim; i++ )
			for( j=0; j<i; j++ )
				Cbet[i][j] = Cbet[j][i];		// symmetric

		// Cbet interpolated prior to Cwit
		for( i=0; i<dim; i++ )
			for( j=0; j<dim; j++ )
				Cbet[i][j] += beta*Cwit[i][j];

		float vari = 0;
		for( i=0; i<dim; i++ )
			vari += Cwit[i][i];
		vari /= dim;

		for( i=0; i<dim; i++ )
			for( j=0; j<dim; j++ )
			{
				if( i==j )
					Cwit[i][j] = (1-beta)*Cwit[i][j]+beta*vari;
				else
					Cwit[i][j] *= 1-beta;
			}
	}

	// Whiten the class mean vectors
	float** P;
	P = theMat.allocMat( dim );
	theMat.diagonalize( Cwit, dim, P );		// P: orthogonal matrix
	for( i=0; i<dim; i++ )
	{
		if( Cwit[i][i]<1E-6 )
			Cwit[i][i] = (float)0.001;
		else
			Cwit[i][i] = (float)sqrt( 1./Cwit[i][i] );	// diagonal
	}

	theMat.matXmat( P, Cwit, dim );		// multiplication saved in P
	theMat.transpose( P, dim, Cwit );	// transpose saved in Cwit for whitening

	float *wmean;
	wmean = new float [classNum*dim];	// whitened means

	for( ci=0; ci<classNum; ci++ )
	{
		for( i=0; i<dim; i++ )
			temp[i] = cmeans[ci*dim+i]-gmean[i];	// shift with gross mean
		for( i=0; i<dim; i++ )
			wmean[ci*dim+i] = innerProd( Cwit[i], temp, dim );
	}

	if( beta>0 )	// whiten the matrix Cbet other than means
	{
		float** Q;
		Q = theMat.allocMat( dim );
		// Cwit is the whitening matrix, P is its transpose
		theMat.matXmat( Cwit, Cbet, dim, Q );
		theMat.matXmat( Q, P, dim, Cbet );

		theMat.freeMat( Q, dim );
	}
	else
	{
		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				Cbet[i][j] = 0;

		for( ci=0; ci<classNum; ci++ )
		{
			for( i=0; i<dim; i++ )
				for( j=i; j<dim; j++ )
					Cbet[i][j] += csnum[ci]*wmean[ci*dim+i]*wmean[ci*dim+j];
		}
		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				Cbet[i][j] /= posiNum-classNum;		// unbiased estimate

		for( i=1; i<dim; i++ )
			for( j=0; j<i; j++ )
				Cbet[i][j] = Cbet[j][i];		// symmetric
	}

	float *phi, *lambda;		// eigenvectors and eigenvalues
	phi = new float [redDim*dim];
	lambda = new float [redDim];

	theMat.PCA( Cbet, dim, phi, lambda, redDim );
	//theMat.characteristic( Cbet, dim, phi, lambda, redDim );

	// transform vectors in the original space
	for( i=0; i<dim; i++ )
		for( j=0; j<redDim; j++ )
			weight[j*dim+i] = innerProd( P[i], phi+j*dim, dim );

	float scale;
	scale = 1/(float)sqrt(lambda[0]+1);	// for transformed data re-scaling
	// the eigenvalue of total covariance equals lambda+1

	delete phi;
	delete lambda;
	delete temp;
	delete wmean;

	theMat.freeMat( Cbet, dim );
	theMat.freeMat( P, dim );
	theMat.freeMat( Cwit, dim );

	delete cmeans;
	delete csnum;

	return scale;
}

// Compute class means and the common covariance matrix
void CTransform::commCova( unsigned char* data, int sampNum, int dim, short* truth, int classNum,float power, float* means, float** Cova, int* csnum )
{
	float* temp;		// intermediate vector of power transformation
	temp = new float [dim];

	int ci, i;
	for( ci=0; ci<classNum; ci++ )
		for( i=0; i<dim; i++ )
			means[ci*dim+i] = 0;



	memset( csnum, 0, classNum*sizeof(int) );	// classwise number of samples
	int n;
	for( n=0; n<sampNum; n++ )
	{
		if( truth[n]<0 )
			continue;

		ci = truth[n];
		csnum[ci] ++;
		powerTrans( data+(int64_t )n*dim, dim, power, temp );
		for( i=0; i<dim; i++ )
			means[ci*dim+i] += temp[i];
	}
	int posiNum = 0;
	for( ci=0; ci<classNum; ci++ )
	{
		posiNum += csnum[ci];
		for( i=0; i<dim; i++ )
			means[ci*dim+i] /= csnum[ci];
	}

	int j;
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			Cova[i][j] = 0;

	// Winthin-class common covaraince
	for( n=0; n<sampNum; n++ )
	{
		if( truth[n]<0 )
			continue;

		//cout<<"n = "<<n<<endl;

		ci = truth[n];
		powerTrans( data+(int64_t )n*dim, dim, power, temp );
		for( i=0; i<dim; i++ )
			temp[i] -= means[ci*dim+i];		// shift w.r.t class mean
		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				Cova[i][j] += temp[i]*temp[j];
	}
	cout<<"sampNum = "<<sampNum<<endl;
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			Cova[i][j] /= posiNum-classNum;		// unbiased estimate


	// covariance matrix is symmetric
	for( i=1; i<dim; i++ )
		for( j=0; j<i; j++ )
			Cova[i][j] = Cova[j][i];

	delete temp;
}

////////////////////////////////////////////////////////////////////////////////////

// Subspace extraction from (positive) sample data by PCA
float CTransform::PCALearn( unsigned char* data, int sampNum, int dim, short* truth,  float power, float* mean, float* phi, int redDim )
{
	int posiNum;		// number of positive samples
	posiNum = grossMean( data, sampNum, dim, truth, power, mean );

	if( sampNum<dim )
		return samplePCA( data, sampNum, dim, power, mean, phi, redDim );

	float* temp;		// intermediate vector of power transformation
	temp = new float [dim];

	float** Cova;			// covariance matrix
	Cova = theMat.allocMat( dim );

	int i, j;
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			Cova[i][j] = 0;
	for( int n=0; n<sampNum; n++ )
	{
		if( truth[n]>=0 )
		{
			powerTrans( data+(int64_t)n*dim, dim, power, temp );
			for( i=0; i<dim; i++ )
				temp[i] -= mean[i];		// shift w.r.t gross mean
			for( i=0; i<dim; i++ )
				for( j=i; j<dim; j++ )
					Cova[i][j] += temp[i]*temp[j];
		}
	}
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			Cova[i][j] /= posiNum-1;	// unbiased estimate

	for( i=1; i<dim; i++ )
		for( j=0; j<i; j++ )
			Cova[i][j] = Cova[j][i];	// symmetric covariance matrix

	float variance, total;
	variance = 0;
	for( i=0; i<dim; i++ )
		variance += Cova[i][i];

	float* lambda;
	lambda = new float [redDim];

	total = theMat.PCA( Cova, dim, phi, lambda, redDim );
	//theMat.characteristic( Cova, dim, phi, lambda, redDim );
	printf( "PCA: %8.1f %8.1f\n", variance, total );

	theMat.freeMat( Cova, dim );

	float scale;
	scale = 1/(float)sqrt(lambda[0]);	// for transformed data re-scaling

	delete temp;

	return scale;
}

// For PCA in the case sampNum<dimension
float CTransform::samplePCA( unsigned char* data, int snum, int dim, float power, float* mean, float* phi, int redDim )
{
	float *tdata;
	tdata = new float [snum*dim];

	int n, i;
	for( n=0; n<snum; n++ )
	{
		powerTrans( data+n*dim, dim, power, tdata+n*dim );
		for( i=0; i<dim; i++ )
			tdata[n*dim+i] -= mean[i];
	}

	float** Cova;			// sample covariance matrix
	Cova = theMat.allocMat( snum );

	int j;
	for( i=0; i<snum; i++ )
		for( j=i; j<snum; j++ )
		{
			Cova[i][j] = theMat.innProduct( tdata+i*dim, tdata+j*dim, dim );
			Cova[i][j] /= snum;
		}
	for( i=1; i<snum; i++ )
		for( j=0; j<i; j++ )
			Cova[i][j] = Cova[j][i];	// symmetric covariance matrix

	float *alpha, *lambda;
	alpha = new float [redDim*snum];
	lambda = new float [redDim];

	float variance, total;
	variance = 0;
	for( i=0; i<snum; i++ )
		variance += Cova[i][i];

	total = theMat.PCA( Cova, snum, alpha, lambda, redDim );
	//theMat.characteristic( Cova, snum, alpha, lambda, redDim );
	printf( "PCA: %8.1f %8.1f\n", variance, total );

	int m;
	for( m=0; m<redDim; m++ )
		for( n=0; n<snum; n++ )
			alpha[m*snum+n] /= sqrt(lambda[m]);

	theMat.freeMat( Cova, snum );

	for( m=0; m<redDim; m++ )
	{
		for( i=0; i<dim; i++ )
		{
			phi[m*dim+i] = 0;
			for( n=0; n<snum; n++ )
				phi[m*dim+i] += alpha[m*snum+n]*tdata[n*dim+i];
		}
		for( i=0; i<dim; i++ )
			phi[m*dim+i] /= sqrt( (float)snum );
	}
	/*float prod;
	for( i=0; i<redDim; i++ )
		for( j=0; j<redDim; j++ )
			prod = theMat.innProduct( phi+i*dim, phi+j*dim, dim );*/

	float scale;
	scale = 1/(float)sqrt(lambda[0]);	// for transformed data re-scaling

	delete alpha;
	delete lambda;
	delete tdata;

	return scale;
}

// Compute the gross mean of positive samples
int CTransform::grossMean( unsigned char* data, int sampNum, int dim, short* truth, 
						   float power, float* mean )
{
	float* temp;		// intermediate vector of power transformation
	temp = new float [dim];

	memset( mean, 0, dim*sizeof(float) );

	int posiNum = 0;		// number of positive samples
	int n, i;
	for( n=0; n<sampNum; n++ )
	{
		if( truth[n]>=0 )
		{
			posiNum ++;
			powerTrans( data+(int64_t)n*dim, dim, power, temp );
			for( i=0; i<dim; i++ )
				mean[i] += temp[i];
		}
	}
	for( i=0; i<dim; i++ )
		mean[i] /= posiNum;

	delete temp;

	return posiNum;
}

