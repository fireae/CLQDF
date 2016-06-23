#include "MQDF_training.h"
#include "MQDF_test.h"
#include "OnlineDef.h"
#include "Matrix.h"
#include <stdio.h>

using namespace std;

float typeMulti;		// multiplier dependent on transformed data type
short codelen;
int classNum;
int ftrDim;
long data_num;
unsigned char *train_ftr;
char *train_labels;
unsigned char *test_ftr;
char *test_labels;
char* codetable;	// table of class codes
short* truth;		// class index numbers

void load_ftr();
void read_training_ftr(string file);
void read_test_ftr(string file);
// The codes are sorted in ascending order in the table
int posInTable( char* label, char* table, int cnum );
// Sort class labels and assign index numbers
int classTable( char* label, int sampNum, char* table, short* truth );
void insertTable( char* label, char* table, int cnum, int pos );
// Save classifier parameters
void saveDictionary(string filetitle);
void loadDictionary(string saveDictionary_file);

CLQDF* pClassifr;
MQDFTEST* Classify_test;
STrParam tp;
void MQDFTrain(char * fname);
void MQDFTest(char * fname);
#define FUNC 2

int main()
{
	int argc = 10;
	char ** argv ;
	argv = new char *[argc];
	for(int i = 0 ; i < argc ; i++)
	{
		argv[i] = new char[1024];
	}

	strcpy(argv[1],"/home/chengcheng/Handwriting/NLPR_Feature_Data");

#if FUNC==1
	MQDFTrain(argv[1]);
#elif FUNC==2
	MQDFTest(argv[1]);
	printf("test is over");

#endif
	//printf("test\n") ;

	//initialize();
	//codelen = 2 ;
	//cout<<codelen<<endl;
	return 0 ;
}

void MQDFTest(char * fname)
{
	//string ftr_file = "/home/chengcheng/Handwriting/NLPR_Feature_Data/HWDB_1_test.mpf";
	string ftr_file = "/home/zhangdanfeng/extr_fea/feature_500000_test.mpf";
	read_test_ftr(ftr_file);

	//pClassifr->trainClassifier(train_ftr,(int64_t)data_num,ftrDim,truth);

	string Dictionary_file = "/home/chengcheng/Handwriting/NLPR_Feature_Data/";
	//Classify_testMQDFTEST(Dictionary_file);

	Classify_test = new MQDFTEST(Dictionary_file);
	Classify_test->testClassifier(test_ftr, (int)data_num, ftrDim, test_labels);

}


void MQDFTrain(char * fname)
{
	//string ftr_file = "/home/chengcheng/Handwriting/NLPR_Feature_Data/HWDB_1_training.mpf";
	string ftr_file = "/home/zhangdanfeng/extr_fea/feature_500000_train.mpf";
	//string ftr_file = "/home/zhangdanfeng/extr_fea/feature_noLoss.mpf";
	read_training_ftr(ftr_file);

	truth = new short [data_num];	// sequential class indices
	codetable = new char [30000*codelen];	// class codes of at most 30000 classes

	// Sort class codes and assign index numbers
	classNum = classTable( train_labels, data_num, codetable, truth );
	printf( "classNum: %d\n", classNum );
	char classify_name[1024];
	strcpy(classify_name,"K40");
	tp.ftrDim = ftrDim;
	tp.power = 0.5;
	tp.redDim = 160;
	tp.iteration = 40;
	tp.classNum = classNum;
	tp.beta = 0 ;
	tp.beFDA = 0 ;
	tp.alpha = 1 ;
	tp.relRate = 0.5;
	tp.pcaPortion = (float)0.95;
	typeMulti = 1 ;

	pClassifr = new CLQDF( tp, classify_name);
	pClassifr->trainClassifier(train_ftr,(int64_t)data_num,ftrDim,truth);

	string saveDictionary_file = "/home/chengcheng/Handwriting/NLPR_Feature_Data/";
	saveDictionary(saveDictionary_file);

}
void load_ftr()
{
	string ftr_file = "/home/chengcheng/Handwriting/NLPR_Feature_Data/HWDB_1_test.mpf";
	read_test_ftr(ftr_file);
}
//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void read_training_ftr(string file)
{
	FILE *fpDat;
	fpDat = fopen(file.c_str(), "rb" );
	if( fpDat==NULL )
	{
		printf( "Cannot open the feature data file.\n" );
		exit( 1 );
	}

	codelen = 2;
	fread( &data_num, sizeof(long), 1, fpDat );
  	fread( &ftrDim, sizeof(int), 1, fpDat );
  	printf("sampNum = %ld, ftrDim = %d\n", data_num, ftrDim);

  	train_ftr = new unsigned char [(int)data_num*ftrDim];
  	train_labels = new char [(int)data_num*codelen];

  	for( int n=0; n<data_num; n++ )
	{
		fread( train_labels+n*codelen, codelen, 1, fpDat );
		fread( train_ftr+n*ftrDim, sizeof(unsigned char), ftrDim, fpDat);
	}
	fclose(fpDat);
}


//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void read_test_ftr(string file)
{
	FILE *fpDat;
	fpDat = fopen(file.c_str(), "rb" );
	if( fpDat==NULL )
	{
		printf( "Cannot open the feature data file.\n" );
		exit( 1 );
	}

	codelen = 2;
	fread( &data_num, sizeof(long), 1, fpDat );
  	fread( &ftrDim, sizeof(int), 1, fpDat );
  	printf("sampNum = %ld, ftrDim = %d\n", data_num, ftrDim);

  	test_ftr = new unsigned char [(int)data_num*ftrDim];
  	test_labels = new char [(int)data_num*codelen];

  	for( int n=0; n<data_num; n++ )
	{
		fread( test_labels+n*codelen, codelen, 1, fpDat );
		fread( test_ftr+n*ftrDim, sizeof(unsigned char), ftrDim, fpDat);
	}
	fclose(fpDat);
}

// Sort class labels and assign index numbers
int classTable( char* label, int sampNum, char* table, short* truth )
{
	int cnum = 0;
	int pos;		// position in code table

	int n;
	for( n=0; n<sampNum; n++ )
	{

		if( label[n*codelen]==-1 )		// outliers sample
			continue;



		pos = posInTable( label+n*codelen, table, cnum );
		//printf("pos = %d ftrDim = %d/sampNum = %ld\n", pos, n, sampNum);
		if( pos==cnum )
		{
			memcpy( table+cnum*codelen, label+n*codelen, codelen );
			cnum ++;
		}
		else if( memcmp(label+n*codelen, table+pos*codelen, codelen) )
		{
			insertTable( label+n*codelen, table, cnum, pos );
			cnum ++;
		}
	}

	for( n=0; n<sampNum; n++ )
	{
		if( label[n*codelen]==-1 )
			truth[n] = -1;
		else
			truth[n] = posInTable( label+n*codelen, table, cnum );
	}
	return cnum;
}

void insertTable( char* label, char* table, int cnum, int pos )
{
	memmove( table+(pos+1)*codelen, table+pos*codelen, (cnum-pos)*codelen );
	memcpy( table+pos*codelen, label, codelen );
}

// The codes are sorted in ascending order in the table
int posInTable( char* label, char* table, int cnum )
{
	if( cnum==0 )
		return 0;

	if( memcmp(label, table, codelen)<=0 )
		return 0;
	else if( memcmp(label, table+(cnum-1)*codelen, codelen)>0 )
		return cnum;

	int b1, b2, t;
	b1 = 0;
	b2 = cnum-1;

	while( b2-b1>1 )
	{
		t = (b1+b2)/2;
		if( memcmp(label, table+t*codelen, codelen)>0 )
			b1 = t;
		else
			b2 = t;
	}
	return b2;
}

void loadDictionary(string saveDictionary_file)
{
	FILE* fp;
	char fname[1024];
	strcpy( fname, saveDictionary_file.c_str());
	strcat( fname, "MQDF.csp" );	// Classifier structure and parameters (CSP)

	fp = fopen( fname, "rb" );

	fread( &codelen, 2, 1, fp );


	fread( &classNum, 4, 1, fp );
	codetable = new char [classNum*codelen];	// class codes of at most 30000 classes

	fread( codetable, codelen, classNum, fp );




}

void saveDictionary(string filetitle)
{
	FILE* fp;
	char fname[100];
	strcpy( fname, filetitle.c_str());
	strcat( fname, "MQDF.csp" );	// Classifier structure and parameters (CSP)
	//fp = fopen( fname, "wb" );
	fp = fopen( "MQDF_500000.csp", "wb" );
	fwrite( &codelen, 2, 1, fp );

	fwrite( &classNum, 4, 1, fp );
	fwrite( codetable, codelen, classNum, fp );

		// transformation and classifier parameters
	pClassifr->saveClassifier( fp );

	fclose( fp );
}
