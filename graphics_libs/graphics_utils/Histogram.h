#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include "tex_ogl.h"

class Histogram {
private:
	int nbBins;
	float equalizationWeight;
	TextureOGL histTex;

public:
	Histogram(int nbBins, float equalizationWeight);
	int computeHistogram(float* tex, int xsize, int ysize);
	TextureOGL& getTexture() {return histTex;}
	int identity();

private:
	// called by computeHistogram
	int updateEqualizationTexture(float* accHistogram);
};

#endif
