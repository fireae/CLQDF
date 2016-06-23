#include "OnlineDef.h"
#include "Matrix.h"
#include <time.h>
#include <fstream>
#include <vector>

// Common functions used by multiple classes, source in Transform.cpp
float innerProd( float*, float*, int );
void powerTrans( unsigned char*, int, float, float* );
float bipolar( float );
void GramSchmidt( float*, int, int, int );
void makeInput( float*, int, float* );	// feature re-scaling, in Transform.cpp



// Procedures for feature transformation (dimensionality reduction)
class CTransform
{
	int grossMean( unsigned char*, int, int, short*, float, float* );
	void commCova( unsigned char*, int, int, short*, int, float, float*, float**, int* );

	CMatrix theMat;

public:
	CTransform();
	~CTransform();

	float dataScale( unsigned char*, int, int, short*, float, float* );
	float transType( float, int );		// transformed data type

	float PCALearn( unsigned char*, int, int, short*, float, float*, float*, int );
	float samplePCA( unsigned char*, int, int, float, float*, float*, int );
	float FisherLearn( unsigned char*, int, int, short*, int, float, float*, float*, int, float );
	float whitenLearn( unsigned char*, int, int, short*, int, float, float*, float*, int );

	void reduceTrans( unsigned char*, int, int, float, float*, float*, int, int, int, float, float* );
	void rescaling( unsigned char*, int, int, float, float*, int, float, float* );
};

// // Quadratic classifiers: MQDF and DLQDF
class CLQDF
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

	CTransform theTrans;
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
	//short codelen;
	//int classNum;
	//char* codetable;	// table of class codes
	//short* truth;		// class index numbers

public:
	CLQDF(STrParam& tp, char* configr);
	~CLQDF();

	void trainClassifier( unsigned char* data, int sampNum, int ftrDim, short* truth);
	void saveClassifier( FILE* );	
	//void classifier_test(unsigned char* data, int sampNum, int ftrDim, char* truth, string dic_file);
};

