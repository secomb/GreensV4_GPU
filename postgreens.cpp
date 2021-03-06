/************************************************************************
postgreens.cpp - analyzes results from greens
Uses parameters from PostGreensParams.dat
Includes problem-specific code from postgreens.cpp.dat
Example of usage: compute survival fraction of cells from drug concentration
TWS, May 2015
**************************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nrutil.h"

void postgreens(void)
{
	extern int max, nsp, nnt, npostgreensparams, npostgreensout,imain;
	extern float **pt, *dtmin, *postgreensparams, *postgreensout;
	char fname[80];
	FILE *ofp;

	sprintf(fname, "Current/PostGreens%03i.ps", imain);
	ofp = fopen(fname, "w");
	//**************************************************************
#include "postgreens.cpp.dat"
//**************************************************************
	fclose(ofp);
}
