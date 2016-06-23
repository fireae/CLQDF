#include "MQDF_test.h"
#include <math.h>

//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MQDFTEST::MQDFTEST(string configr)
{
	FILE* fp;
	char fname[1024];
	strcpy( fname, configr.c_str());
	strcat( fname, "MQDF.csp" );	// Classifier structure and parameters (CSP)

	//fp = fopen( fname, "rb" );
	fp = fopen( "MQDF_500000.csp", "rb" );
	fread( &codelen, 2, 1, fp );


	fread( &classNum, 4, 1, fp );
	codetable = new char [classNum*codelen];	// class codes of at most 30000 classes

	fread( codetable, codelen, classNum, fp );
	
	fread( &ftrDim, 4, 1, fp );
	fread( &power, sizeof(float), 1, fp );
	fread( transcode, 20, 1, fp );
	fread( &redDim, 4, 1, fp );
	fread( &residual, 4, 1, fp );

	// Transformation parameters
	gmean = new float [ftrDim];			// gross mean vector
	fread( gmean, sizeof(float), ftrDim, fp );
	trbasis = new float [redDim*ftrDim];	// transformation weight vectors
	//typeMulti = 1;
	fread( trbasis, sizeof(float), redDim*ftrDim, fp );
	fread( &bipol, sizeof(int), 1, fp );
	fread( &dscale1, sizeof(float), 1, fp );

	// Classifier configuration
	char clasfstr[20] = "MQDF";
	char conf[20];
	fread( clasfstr, 20, 1, fp );
	fread( conf, 20, 1, fp );



	// dictionary

	fread( &classNum, sizeof(int), 1, fp );
	fread( &redDim, sizeof(int), 1, fp );
	knum = new int [classNum];
	fread( knum, sizeof(int), classNum, fp );
	means = new float [classNum*redDim];		// class mean vectors
	fread( means, sizeof(float), classNum*redDim, fp );



	phi = new float* [classNum];
	lambda = new float* [classNum];
	for( int ci=0; ci<classNum; ci++ )
	{
		phi[ci] = new float [ knum[ci]*redDim ];	// principal eigenvectors
		fread( phi[ci], sizeof(float), knum[ci]*redDim, fp );
	}

	loglambda = new float [classNum];
	for( int ci=0; ci<classNum; ci++ )
	{
		//cout<<"ci = "<<ci  <<"knum[ci] = "<<knum[ci]<<endl;
		lambda[ci] = new float [ knum[ci]+1 ];		// principal eigenvalues
		fread( lambda[ci], sizeof(float), knum[ci]+1, fp );
		loglambda[ci] = (redDim-knum[ci])*log( lambda[ci][knum[ci]] );
		for( int j=0; j<knum[ci]; j++ )
			loglambda[ci] += log( lambda[ci][j] );
	}

	transform = 2;
	rankN = 10;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------
// Power transformation of one feature vector
void MQDFTEST::powerTrans( unsigned char* vect, int dim, float power, float* fvect )
{
	for( int i=0; i<dim; i++ )
	{
		if( power==1 )
			fvect[i] = (float)vect[i];
		else if( power==0.5 )
		{
			if( vect[i]<0 )
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
//-----------------------------------------------------------------------------------------------------------------------------------------------
float MQDFTEST::bipolar( float a )
{
	float exv2;
	exv2 = (float)exp( -2*a );
	return (1-exv2)/(1+exv2);
}
// Retrieve class index from code table (sorted in ascending order)
int MQDFTEST::posInTable( char* label )
{
	if( memcmp(label, codetable, codelen)<=0 )
		return 0;
	else if( memcmp(label, codetable+(classNum-1)*codelen, codelen)>0 )
		return classNum;

	int b1, b2, t;
	b1 = 0;
	b2 = classNum-1;

	while( b2-b1>1 )
	{
		t = (b1+b2)/2;
		if( memcmp(label, codetable+t*codelen, codelen)>0 )
			b1 = t;
		else
			b2 = t;
	}

	return b2;
}

// Feature transformation to PCA, whitening or Fisher subspace
void MQDFTEST::featureTrans( unsigned char* ftr, int dim, float* input )
{
	float *shx;
	shx = new float [dim];

	powerTrans( ftr, dim, power, shx );

	int i;
	for( i=0; i<dim; i++ )
		shx[i] -= gmean[i];		// shift with respect to gross mean

	int j;
	float proj, euclid;

	if( residual )
	{
		euclid = 0;
		for( i=0; i<dim; i++ )
			euclid += shx[i]*shx[i];
	}

	for( j=0; j<redDim; j++ )
	{
		proj = innerProd( shx, trbasis+j*dim, dim );
		if( residual )
			euclid -= proj*proj;

			input[j] = proj*dscale1 ;
	}
	if( residual )
		input[redDim] = euclid*dscale1*dscale1 ;


	delete []shx;
}
float MQDFTEST::innerProd( float* v1, float* v2, int dim )
{
	float sum = 0;
	for( int i=0; i<dim; i++ )
		sum += v1[i]*v2[i];

	return sum;
}
//-----------------------------------------------------------------------------------------------------------------------------------------------
void MQDFTEST::testClassifier( unsigned char* data, int sampNum, int ftrDim, char* labels)
{
	unsigned char *ftr = new unsigned char[ftrDim];
	float eudist[rankN1];
	float* input = new float [redDim];
	short index[RankNum] ;
	short preIdx[rankN1] ;
	float output[RankNum];
	int topCorrect = 0 ,rankcorrect = 0 ;

	for(int n = 0; n < sampNum; n++)
	{
		int cls = posInTable( labels+n*codelen );
		featureTrans( data+n*ftrDim, ftrDim, input );

		if(cls < 0) continue ;
		//cout<<"cls = "<<cls<<endl;
		nearSearch(input, redDim, eudist, preIdx, rankN1);
		MQDF( input, redDim, eudist, preIdx, rankN1, output,  index, RankNum );
		if(index[0] == cls)topCorrect++;
		for(int j = 0 ; j < RankNum ; j++)
		{
			if(index[j] == cls)
			{
				rankcorrect++;
					break;
			}
		}
		if(n%30000==0&&n!=0)
		{
			printf( "top correct: %6.2f, rank %d correct %6.2f\n", 100.*topCorrect/(n+1), RankNum, 100.*rankcorrect/(n+1) );
		}
	}
	//featureTrans( data, ftrDim, input );

	cout<<"sampNum = "<<sampNum<<endl;
	cout<<"redDim = "<<redDim<<endl;


}

void MQDFTEST::MQDF( float* ftr, int dim, float* eudist, short* preIdx, int rank1, float* qdmin, short* index, int rankNum )
{
	int k;
	for( k=0; k<rankNum; k++ )
		qdmin[k] = (float)1E12+k;

	float euclid;
	float* shx;
	shx = new float [dim];

	int cls, i, m;
	float proj;
	float qdist, tdist;
	int pos, kt;
	for( int ri=0; ri<rankN1; ri++ )
	{
		if( preIdx )
			cls = preIdx[ri];
		else
			cls = ri;

		for( i=0; i<dim; i++ )
			shx[i] = (float)ftr[i]-means[cls*dim+i];

		if( preIdx )
			euclid = eudist[ri];
		else
		{
			euclid = 0;
			for( i=0; i<dim; i++ )
				euclid += shx[i]*shx[i];
		}

		qdist = loglambda[cls];
		kt = knum[cls];		// truncated knum
		if( kt==dim )
			kt -= 1;
		for( m=0; m<kt; m++ )
		{
			proj = innerProd( shx, phi[cls]+m*dim, dim );
			euclid -= proj*proj;
			qdist += proj*proj/lambda[cls][m];
			tdist = qdist+euclid/lambda[cls][m+1];	// increasing sequence
			if( tdist>=qdmin[rankNum-1] )
				break;
		}
		qdist = tdist;

		if( qdist<qdmin[rankNum-1] )
		{
			pos = posAscd( qdist, qdmin, rankNum );
			for( k=rankNum-1; k>pos; k-- )
			{
				qdmin[k] = qdmin[k-1];
				index[k] = index[k-1];
			}
			qdmin[pos] = qdist;
			index[pos] = cls;
		}
	}

	delete shx;
}

void MQDFTEST::nearSearch( float* input, int dim, float* dmin, short* index, int rankNum )
{
	int ri;
	for( ri=0; ri<rankNum; ri++ )
		dmin[ri] = (float)1E12+ri;

	float dist, diff;
	int ci, i;
	int pos;

	for( ci=0; ci<classNum; ci++ )
	{
		dist = 0;
		for( i=0; i<dim; i++ )
		{
			diff = (float)input[i]-means[ci*dim+i];
			//float T = means[ci*dim+i] ;
			//float S = (REAL)input[i] ;
			//printf("%3.8f ,         %3.8f\n",T,S);
			dist += diff*diff;		// square Euclidean distance
			if( dist>=dmin[rankNum-1] )
				break;
		}

		if( dist<dmin[rankNum-1] )
		{
			pos = posAscd( dist, dmin, rankNum );
			for( ri=rankNum-1; ri>pos; ri-- )
			{
				dmin[ri] = dmin[ri-1];
				index[ri] = index[ri-1];
			}
			dmin[pos] = dist;
			index[pos] = ci;
		}
	}
}

// Rank position in an ordered array, by bisection search
int MQDFTEST::posAscd( float dist, float* dmin, int candiNum )
{
	if( dist<dmin[0] || candiNum<=1 )
		return 0;

	int b1, b2, pos;

	b1 = 0;
	b2 = candiNum-1;
	while( b2-b1>1 )	// bi-section search
	{
		pos = (b1+b2)/2;
		if( dist<dmin[pos] )
			b2 = pos;
		else
			b1 = pos;
	}
	return b2;
}









