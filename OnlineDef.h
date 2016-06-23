/* Online character recognition definition */
#include <string>
#include  <stdio.h>
#include  <stdlib.h> 
#include  <vector>
#include  <fstream>
#include  <sstream> 
#include  <math.h>
#include  <iostream> 
#include  <string.h>
#include <fstream>  // NOLINT(readability/streams)
//#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <iostream>
//#include <utility>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <stdint.h>
#include <iomanip> //
using namespace std;

#ifndef	OLCRDEF_H	
#define	OLCRDEF_H


#define RankNum 10
#define rankN1 200

// structure of training parameters
struct STrParam {
	int ftrDim;
	float power;
	int transform;
	float beta;
	float beFDA;
	int redDim;
	int iteration;
	int classNum;

	float alpha;		// weight decay coefficient
	float relRate;		// relative learning rate
	float pcaPortion;
};


const float FeatureLevel = 35.0;
//---------------------------------------------------------------------------
// coordinate short
typedef struct {
	short x;
	short y;
} SPoint;
//---------------------------------------------------------------------------
// coordinate float
typedef struct {
	float x;
	float y;
} SFPoint;
//---------------------------------------------------------------------------
typedef struct{
	int maxx, maxy, minx, miny;
} PAT_POS;


#endif /* OLCRDEF_H */
