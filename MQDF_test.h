#include "OnlineDef.h"
#include "Matrix.h"
#include <time.h>
#include <fstream>
#include <vector>


// // Quadratic classifiers: MQDF and DLQDF
class MQDFTEST
{
protected:
	int classNum;
	int* knum;		// number of principal eigenvectors, class specific
	int kmax;
	float pcaPortion;	// percentage of variance

	float* means;		// classwise mean vectors
	float variance;		// global variance

	float** phi;			// classwise eigenvectors
	float** lambda;		// lambda[0~k], lambda[k]=delta
	float* cthresh;	// class-wise thresholds
	float *coTrace, *coRemn;		// covariance traces and remains

	float* coarse;		// cluster centers for coarse classification
	int coarNum, coarRank;
	short** coarSet;	// prototypes in each coarse cluster
	int* coarSize;		// number of prototypes in each cluster
	short* coarLabel;	// cluster label of prototypes

	int initial;	// initialization from dictionary file
	int method;		// k=Kimura, q=average, r=regularized DA, c,d,e=discriminative

	//CTransform theTrans;
	void learnTransform( unsigned char* data, int sampNum, int ftrDim, short* truth);
	void dataTransform( unsigned char* data, int sampNum, int ftrDim, int bipol, float* trData );


	void LQDFtrain( float* data, int sampNum, int dim, short* truth, float regu1,float alpha, int iteration, float relRate, float pcaPortion);
	void meanAll( float*, int, int, short*, float*, int* );
	float covariance( float*, int, int, float*, float** );
	float commCova( float*, int, int, short*, float*, float** );
	void classKLT( float* data, int sampNum, int dim, short* truth, float regu1, float portion );
	void interpolate( float**, float**, int, float );
	int samplePCA( float*, int, int, float*, float*, float*, int, float, float&, float& );
	void initLQDF( float*, int, int, short*, float, int, float );
	void initLambda( float );
	void allocLQDF();
	char transcode[20];	// transformation code

	int ftrDim;
	float power;			// power of variable transformation
	float* gmean;		// gross mean vector
	float* trbasis;		// transformation basis
	int iteration;		// number of data sweeps in learning
	float beta;			// regularization of common covariance
	float beFDA;			// interpolation parameter for FDA
	int redDim;		// reduced and enhanced dimensionality
	float dscale1;			// for feature re-scaling
	float relRate;		// relative learning rate
	float alpha;			// regularization of within-class distance
	int bipol;			// bipolar transform or not
	int residual;		// projection residual to be used or not
	char configr[20];	// classifier configuration

	//void load_dic_file(string dic_file);
	short codelen;
	//int classNum;
	char* codetable;	// table of class codes
	//short* truth;		// class index numbers
	float* loglambda;

	int transform;
	//float eudist;
	int rankN;
	// Power transformation of one feature vector
	void powerTrans( unsigned char* vect, int dim, float power, float* fvect );
	int posInTable( char* label );
	void featureTrans( unsigned char* ftr, int dim, float* input );
	float bipolar( float a );
	float innerProd( float* v1, float* v2, int dim );
	void nearSearch( float* input, int dim, float* dmin, short* index, int rankNum );
	int posAscd( float dist, float* dmin, int candiNum );
	void MQDF( float* ftr, int dim, float* eudist, short* preIdx, int rank1, float* qdmin, short* index, int rankNum );

public:
	MQDFTEST(string configr);
	~MQDFTEST();

	void testClassifier( unsigned char* data, int sampNum, int ftrDim, char* labels);
	void saveClassifier( FILE* );	
	//void classifier_test(unsigned char* data, int sampNum, int ftrDim, char* truth, string dic_file);
};
