#include <stdio.h>
#include "Histogram.h"
#include "tex_ogl.h"
#include "glincludes.h"
#include "Array3D.h"

//----------------------------------------------------------------------
Histogram::Histogram(int nbBins, float equalizationWeight)
{
	this->nbBins = nbBins;
	this->equalizationWeight = equalizationWeight;
	this->identity();

	// initial histogram should be the identity
}
//----------------------------------------------------------------------
int Histogram::updateEqualizationTexture(float* accHistogram)
{
	int i, j;
	int ix;

	Array3D h(4, nbBins, 2);
	printf("----- updateEqualizationTexture\n");

	for (j=0; j < 2; j++) {
	for (i=0; i < nbBins; i++) {
		h(0,i,j) = accHistogram[i];
		h(1,i,j) = accHistogram[i];
		h(2,i,j) = accHistogram[i];
		h(3,i,j) = 1.0;
		if (j == 0) printf("update accHistogram[%d]= %f\n", i, accHistogram[i]);
	}}


	histTex.init_targ(nbBins, 2, TARGET); // should work with 1
	histTex.bind();
	// 1 component
    //histTex.load(GL_LUMINANCE, GL_LUMINANCE, GL_FLOAT, h.getDataPtr()); // 1 component
    histTex.load(FLOAT_BUFFER, GL_RGBA, GL_FLOAT, h.getDataPtr()); // 1 component

	return 0;
}
//----------------------------------------------------------------------
int Histogram::identity()
{
	float accHistogram[nbBins]; // = new double [nbBins];

	float dx = 1. / (nbBins-1);
	for (int i=0; i < nbBins; i++) {
		accHistogram[i] = i*dx;
	}
	printf("*** INSIDE identity, call update\n");
	updateEqualizationTexture(accHistogram);

	//delete [] accHistogram;
}
//----------------------------------------------------------------------
int Histogram::computeHistogram(float* tex, int xsize, int ysize)
// xsize, ysize: size of texture
// equalizationWeight = 1: full equalization, = 0: identiy
// Assume we are dealing with a RGBA texture. Only sample 1st component (we assume
// a grey-scale texture
{
	printf("****** computeHistogram ***** \n");
	float* hist = new float [nbBins];
	float* c_hist = new float [nbBins];
	int i, j;

	// subset of texture to sample. Of course I should check that the texture is at least
	// as large as 256  (TODO)

	printf("MAKE SURE THAT TEXTURES > 256x256\n");

	int nx = 256;
	int ny = 256;

	if (nx < xsize) nx = xsize;
	if (ny < ysize) ny = ysize;

	float npts = 1. / ((float) nx*ny); // histogram based on the first 256^2 points in the domain

	for (i=0; i < nbBins; i++) {
		hist[i] = 0;
	}

	if (tex == NULL) {
		for (i=0; i < nbBins; i++) {
			hist[i] = (double) i;
		}
	} else {
		float nbf = (float) (nbBins-1.);
		for (j=0; j < ny; j++) {
			float* row = tex + j*4*xsize;
			//printf("%d, row= %ld\n", j, (long) row);
		for (i=0; i < nx; i++) {
			float value = *row;
			row += 4; // RGBA
			//printf("%d, %d, row= %ld\n", i, j, (long) row);
			float bin = nbf * value; // gray is between zero and 1
			//printf("value= %f, bin= %f\n", value, bin);
			hist[(int) bin]++; // update histogram's bin
		}}
	}

	// Accumulated histogram
	//printf("accumulate\n");

	c_hist[0] = 0.0;
	for (i=1; i < nbBins; i++) {
		c_hist[i] = c_hist[i-1] + hist[i];
	}

	double sum = c_hist[nbBins-1];
	//normalize
	for (i=0; i < nbBins; i++) {
		c_hist[i] /= sum;
		printf("c_hist[%d]= %f\n", i, c_hist[i]);
		double id = ((double) i) / nbBins;
		c_hist[i] = equalizationWeight*c_hist[i] + (1.0-equalizationWeight)*id;
		printf("       weighted c_hist[%d]= %f\n", i, c_hist[i]);
		//c_hist[i] = 0.0;
	}

	printf("*** INSIDE computeHistogram, call update\n");
	updateEqualizationTexture(c_hist);

	delete [] hist;
	delete [] c_hist;

	return 0;
}
//----------------------------------------------------------------------
