#include "MQDF_training.h"
#include <math.h>

char transcode[20] = "DEF&FDA";
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CLQDF::CLQDF(STrParam& tp, char* configr)
{
	ftrDim = tp.ftrDim ;
	power = tp.power ;
	redDim = tp.redDim ;
	iteration = tp.iteration ;
	classNum = tp.classNum ;
	alpha = tp.alpha;
	relRate = tp.relRate;
	beta = tp.beta ;
	beFDA = tp.beFDA ;
	pcaPortion = tp.pcaPortion;
	bipol = 0 ;
	residual =  0 ;

	if( configr[0]=='t' || configr[0]=='T' )	// truncated eigenvalues
	{
		method = 1;
		printf( "MQDF: truncated minor eigenvalues\n" );
	}
	else if( configr[0]=='k' || configr[0]=='K' )	// Kimura, constant minor eigenvalue 
	{
		method = 2;
		printf( "MQDF: constant minor eigenvalue\n" );
	}
	else if( configr[0]=='r' || configr[0]=='R' )	// regularization
	{
		method = 3;
		printf( "MQDF: regularized covariance matrices\n" );
	}
	else if(configr[0]=='n' || configr[0]=='N')// Kimura, constant minor eigenvalue ,coding by Nakagawa laboratory
	{
		method = 8;
		printf( "MQDF: constant minor eigenvalue, coding by Nakagawa laboratory\n" );
	}
	else
	{
		method = 0;		// average minor eigenvalue
		printf( "MQDF: average minor eigenvalues\n" );
	}

	kmax = atoi( configr +1);	// maximum number of eigenvectors
	if( kmax>redDim )
		kmax = redDim;

	allocLQDF();
	initial = 0;
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::allocLQDF()
{
	trbasis = new float [redDim*ftrDim];	// transformation basis
	gmean = new float [ftrDim];		// gross mean, always used

	//cout<<"ftrDim = "<<ftrDim<<endl;
	//cout<<"classNum = "<<classNum<<endl;

	knum = new int [classNum];
	means = new float [classNum*redDim];		// class mean vectors
	phi = new float* [classNum];
	lambda = new float* [classNum];
	for( int ci=0; ci<classNum; ci++ )
	{
		phi[ci] = new float [kmax*redDim];		// class-specific eigenvectors
		lambda[ci] = new float [kmax+1];		// principal eigenvalues
	}
	cthresh = new float [classNum];
	coTrace = new float [classNum];		// trace of covariance matrix
	coRemn = new float [classNum];		// remaining trace in minor subspace
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CLQDF::~CLQDF()
{
	delete []trbasis;
	delete []gmean;
	delete []means;
	delete []knum;

	for( int ci=0; ci<classNum; ci++ )
	{
		delete []phi[ci];
		delete []lambda[ci];
	}
	delete []phi;
	delete []lambda;
	delete []cthresh;
	delete []coTrace;
	delete []coRemn;
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::saveClassifier( FILE* fp )// Save classifier parameters
{
	fwrite( &ftrDim, 4, 1, fp );
	fwrite( &power, sizeof(float), 1, fp );
	fwrite( transcode, 20, 1, fp );
	fwrite( &redDim, 4, 1, fp );
	fwrite( &residual, 4, 1, fp );

	// Transformation parameters
	fwrite( gmean, sizeof(float), ftrDim, fp );
	fwrite( trbasis, sizeof(float), redDim*ftrDim, fp );
	fwrite( &bipol, sizeof(int), 1, fp );
	fwrite( &dscale1, sizeof(float), 1, fp );

	// Classifier configuration
	char clasfstr[20] = "MQDF";
	fwrite( clasfstr, 20, 1, fp );
	fwrite( configr, 20, 1, fp );

	// dictionary 

	fwrite( &classNum, sizeof(int), 1, fp );
	fwrite( &redDim, sizeof(int), 1, fp );
	fwrite( knum, sizeof(int), classNum, fp );
	fwrite( means, sizeof(float), classNum*redDim, fp );

	for( int ci=0; ci<classNum; ci++ )
		fwrite( phi[ci], sizeof(float), knum[ci]*redDim, fp );
	for( int ci=0; ci<classNum; ci++ )
		fwrite( lambda[ci], sizeof(float), knum[ci]+1, fp );

}

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::learnTransform( unsigned char* data, int sampNum, int ftrDim, short* truth)
{

	dscale1 = theTrans.FisherLearn( data, sampNum, ftrDim, truth,	classNum, power, gmean, trbasis, redDim, beFDA);
	printf( "Fisher discriminant analysis: %d-->%d\n", ftrDim, redDim );

}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::dataTransform( unsigned char* data, int sampNum, int ftrDim, int bipol, float* trData )
{
	theTrans.reduceTrans( data, sampNum, ftrDim, power, gmean, trbasis,redDim, residual, bipol, dscale1, trData );
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::trainClassifier( unsigned char* data, int sampNum, int ftrDim, short* truth)
{
	learnTransform( data, sampNum, ftrDim, truth);
	float* trData;
	trData = new float [(int64_t)sampNum*redDim];
	if( trData==0 )		printf( "Invalid memory for trData\n" );
	dataTransform( data, sampNum, ftrDim, bipol, trData );

	LQDFtrain( trData, sampNum, redDim, truth, beta, alpha, iteration, relRate, pcaPortion );

	delete []trData;
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Smooth class-specific covariance with the common covariance
void CLQDF::interpolate( float** Cova, float** Comm, int dim, float coef )
{
	int i, j;
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			Cova[i][j] = (1-coef)*Cova[i][j]+coef*Comm[i][j];

	for( i=1; i<dim; i++ )
		for( j=0; j<i; j++ )
			Cova[i][j] = Cova[j][i];
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// For PCA in the case sampNum<dimension
int CLQDF::samplePCA( float* data, int snum, int dim, float* mean, float* phi, float* lambda, int redDim, float pcaPortion, float& vari, float& accumu )
{
	float *tdata;
	tdata = new float [snum*dim];

	int n, i;
	for( n=0; n<snum; n++ )
		for( i=0; i<dim; i++ )
			tdata[n*dim+i] = (float)data[n*dim+i]-mean[i];

	CMatrix theMat;
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

		float *alpha;
		alpha = new float [redDim*snum];

		float total;
		vari = 0;
		for( i=0; i<snum; i++ )
			vari += Cova[i][i];

		total = theMat.PCA( Cova, snum, alpha, lambda, redDim );
		//theMat.characteristic( Cova, snum, alpha, lambda, redDim );

		int pcaDim;
		float thresh;
		thresh = vari*pcaPortion;	// percentage of variance
		j = 0;
		accumu = lambda[0];
		while( accumu<thresh && j<redDim-1 )
		{
			j ++;
			accumu += lambda[j];
		}
		pcaDim = j+1;

		printf( "PCA: %8.2f, %8.2f, %d\n", vari, total, pcaDim );

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

		delete alpha;
		delete tdata;

		return pcaDim;
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::LQDFtrain( float* data, int sampNum, int dim, short* truth, float regu1,float alpha, int iteration, float relRate, float pcaPortion )
{
	if( initial==0 )	initLQDF( data, sampNum, dim, truth, regu1, alpha, pcaPortion );
	else
	{
		int totalPC = 0;
		for( int ci=0; ci<classNum; ci++ )
			totalPC += knum[ci];
		printf( "Principal eigenvectors: %6.2f\n", (float)totalPC/classNum );
	}

	int rankNum;	// for accelerating rival class search
	if( classNum<25 )
		rankNum = classNum;
	else
		rankNum = int( 2*sqrt((float)classNum) )+15;
	initial++;
}
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void CLQDF::initLQDF( float* data, int sampNum, int dim, short* truth, float regu1, int hp, float pcaPortion )
{
	float regu2;

	classKLT( data, sampNum, dim, truth, regu1, pcaPortion );


	if( method==2 )
	{
		regu2 = (float)0.1*(hp+1);		// multiplier of average variance
		printf( "delta: %5.2f times average eigenvalue\n", regu2 );
	}
	else
	{
		regu2 = (float)0.1*hp;	// regularization coefficient of identity matrix
		printf( "regularization coefficient: %5.2f\n", regu2 );
	}

	initLambda( regu2 );	// set the common minor eigenvalues

	for( int ci=0; ci<classNum; ci++ )
		cthresh[ci] = 0;
}

// K-L transform of class-wise covariance matrices
void CLQDF::classKLT( float* data, int sampNum, int dim, short* truth, float regu1, float portion )
{
	int* csnum;		// number of samples of each class
	csnum = new int [classNum];

	meanAll( data, sampNum, dim, truth, means, csnum );

	CMatrix* pMat;
	pMat = new CMatrix;

	float **Cova, **Comm;
	float poolVar;
	Cova = pMat->allocMat( dim );

	// Compute the common covariance matrix
	if( regu1>0 )
	{
		Comm = pMat->allocMat( dim );
		poolVar = commCova( data, sampNum, dim, truth, means, Comm );
	}

	float* tdata;		// feature data of a class
	int tnum, n;
	float total, error;
	float accumu, thresh;	// accumulated variance
	int m;

	variance = 0;
	for( int ci=0; ci<classNum; ci++ )
	{
		// Collect the data of one class
		tdata = new float [ csnum[ci]*dim ];
		tnum = 0;
		for( n=0; n<sampNum; n++ )
		{
			if( truth[n]==ci )
			{
				memcpy( tdata+(int64_t)tnum*dim, data+(int64_t)n*dim, dim*sizeof(float) );
				tnum ++;
			}
		}

		if( tnum<dim )
		{
			printf( "class %d: %d, ", ci, tnum );
			knum[ci] = samplePCA( tdata, tnum, dim, means+ci*dim, phi[ci], lambda[ci], kmax, portion, coTrace[ci], accumu );
		}
		else
		{
			coTrace[ci] = covariance( tdata, tnum, dim, means+ci*dim, Cova );

			// Regulate class covariance with common covariance
			if( regu1>0 )
			{
				interpolate( Cova, Comm, dim, regu1 );
				coTrace[ci] = (1-regu1)*coTrace[ci]+regu1*poolVar;
			}

			total = pMat->PCA( Cova, dim, phi[ci], lambda[ci], kmax );
			error = pMat->characteristic( Cova, dim, phi[ci], lambda[ci], kmax );

			thresh = coTrace[ci]*portion;	// percentage of variance
			m = 0;
			accumu = lambda[ci][0];
			while( accumu<thresh && m<kmax-1 )
			{
				m ++;
				accumu += lambda[ci][m];
			}
			knum[ci] = m+1;
			printf( "class %2d, %d: %8.2f, %8.2f, %8.6f, %d\n", ci, tnum, coTrace[ci], total, error, knum[ci] );
		}
		variance += coTrace[ci];
		coRemn[ci] = coTrace[ci]-accumu;	// remaining varaince in minor subspace

		delete tdata;
	}
	variance /= classNum;

	pMat->freeMat( Cova, dim );
	if( regu1>0 )
		pMat->freeMat( Comm, dim );

	delete csnum;
	delete pMat;
}

// Calculate the covariance matrix of the sample data, given the mean vector
float CLQDF::covariance( float* data, int tnum, int dim, float* mean, float** C )
{
	float* shx;		// shifted vector with respect to class mean
	shx = new float [dim];

	int i, j;
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			C[i][j] = 0;

	for( int n=0; n<tnum; n++ )
	{
		for( i=0; i<dim; i++ )
			shx[i] = (float)data[(int64_t)n*dim+i]-mean[i];

		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				C[i][j] += shx[i]*shx[j];
	}
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			C[i][j] /= tnum-1;		// unbiased estimate

	// covariance matrix is symmetric
	for( i=1; i<dim; i++ )
		for( j=0; j<i; j++ )
			C[i][j] = C[j][i];		// symmetric

	delete shx;

	float trace = 0;
	for( i=0; i<dim; i++ )
		trace += C[i][i];

	return trace;
}

// Common covariance matrix on class-specific means
float CLQDF::commCova( float* data, int sampNum, int dim, short* truth, float* means, float** C )
{
	float* shx;		// shifted vector with respect to class mean
	shx = new float [dim];

	int i, j;
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			C[i][j] = 0;

	int ci;
	int posiNum = 0;
	for( int n=0; n<sampNum; n++ )
	{
		ci = truth[n];
		if( ci<0 )
			continue;

		posiNum ++;
		for( i=0; i<dim; i++ )
			shx[i] = (float)data[(int64_t)n*dim+i]-means[ci*dim+i];
		for( i=0; i<dim; i++ )
			for( j=i; j<dim; j++ )
				C[i][j] += shx[i]*shx[j];
	}
	for( i=0; i<dim; i++ )
		for( j=i; j<dim; j++ )
			C[i][j] /= posiNum-classNum;		// unbiased estimate

	// covariance matrix is symmetric
	for( i=1; i<dim; i++ )
		for( j=0; j<i; j++ )
			C[i][j] = C[j][i];

	delete shx;

	float vari = 0;
	for( i=0; i<dim; i++ )
		vari += C[i][i];

	return vari;
}

// Compute the mean vectors of all classes
void CLQDF::meanAll( float* data, int sampNum, int dim, short* truth,float* means, int* csnum )
{
	int ci, i;
	for( ci=0; ci<classNum; ci++ )
		for( i=0; i<dim; i++ )
			means[ci*dim+i] = 0;

	memset( csnum, 0, classNum*sizeof(int) );

	for( int n=0; n<sampNum; n++ )
	{
		ci = truth[n];
		if( ci>=0 )
		{
			csnum[ci] ++;
			for( i=0; i<dim; i++ )
				means[ci*dim+i] += data[(int64_t)n*dim+i];
		}
	}
	for( ci=0; ci<classNum; ci++ )
		for( i=0; i<dim; i++ )
			means[ci*dim+i] /= csnum[ci];
}

// Set the common minor eigenvalues
void CLQDF::initLambda( float regu2 )
{
	int ci, m;
	float vari, remv;
	if( method==3 )		// regularization
	{
		for( ci=0; ci<classNum; ci++ )
		{
			vari = coTrace[ci]/redDim;
			for( m=0; m<knum[ci]; m++ )		// covariance matrix smoothing
				lambda[ci][m] = (1-regu2)*lambda[ci][m]+regu2*vari;
		}
	}

	if( method==2 ||method == 8)		// constant minor eigenvalue
	{
		float delta;
		vari = variance/redDim;
		delta = regu2*vari;				// regu2 is multiplier in this case
		for( ci=0; ci<classNum; ci++ )
			lambda[ci][ knum[ci] ] = delta;
	}
	else if( method==3 )
	{
		for( ci=0; ci<classNum; ci++ )
		{
			vari = coTrace[ci]/redDim;
			if( knum[ci]<redDim )
				remv = coRemn[ci]/(redDim-knum[ci]);
			else
				remv = 0;
			lambda[ci][ knum[ci] ] = (1-regu2)*remv+regu2*vari;
		}
	}
	else if( method==1 )	// truncated eigenvalue
	{
		float aver=0;
		for( ci=0; ci<classNum; ci++ )
			aver += lambda[ci][ knum[ci]-1 ];
		aver /= classNum;
		for( ci=0; ci<classNum; ci++ )
			lambda[ci][ knum[ci] ] = aver;//lambda[ci*(knum+1)+knum-1];
	}
	else
	{
		for( ci=0; ci<classNum; ci++ )
		{
			if( knum[ci]<redDim )
				remv = coRemn[ci]/(redDim-knum[ci]);	// average minor eigenvalue
			else
				remv = 0;
			lambda[ci][ knum[ci] ] = remv;
		}
	}

	for( ci=0; ci<classNum; ci++ )
	{
		for( m=0; m<=knum[ci]; m++ )
		{
			if( m<knum[ci] )
			{
				if( lambda[ci][m]<lambda[ci][ knum[ci] ] )
					lambda[ci][m] = lambda[ci][ knum[ci] ];
			}
			if( lambda[ci][m]<1E-6 )	// singular
				lambda[ci][m] = (float)1E-6;
		}
	}
}

