#include "utils.h"
#include "log.h"
#include <fstream>
#include <cstdlib>
using namespace std;
using namespace isph;

const double Consts::Pi = 3.141592653589793238462643;
const double Consts::PiInv = 1.0/Consts::Pi;
const double Consts::e = 2.7182818284590452354;
const double Consts::eInv = 1.0/Consts::e;
const double Consts::g = 9.80665;
const double Consts::Water::StdDensity = 998.2071;
const double Consts::Water::StdViscosity = 0.001002;
const double Consts::Sea::StdDensity = 1025;
const double Consts::Sea::StdViscosity = 0.00108;

double Consts::Water::Density(int temperature)
{
	return 0.14395 / pow(0.0112, 1 + pow(1-(temperature+273)/649.727, 0.05107));
}

double Consts::Water::Viscosity(int temperature)
{
	return exp(-3.7188 + 578.919 / (135.454 + temperature)) / 1000;
}

double Consts::Sea::Density(int temperature)
{
	return StdDensity;
}

double Consts::Sea::Viscosity(int temperature)
{
	return StdViscosity;
}

unsigned int Utils::NearestPowerOf2( unsigned int num )
{
	num--;
	num |= num >> 1;
	num |= num >> 2;
	num |= num >> 4;
	num |= num >> 8;
	num |= num >> 16;
	num++;
	return num;
}

unsigned int Utils::NearestMultiple( unsigned int num, unsigned int divisor, bool snapUp )
{
	if((num % divisor) == 0)
		return num;
	else if(snapUp)
		return num - (num % divisor) + divisor;
	else
		return num - (num % divisor);
}

std::string Utils::LoadCLSource( const std::string& filename )
{
	std::string text;

	// open archive of subprograms
	std::ifstream stream("isph.cta", std::ios_base::binary|std::ios_base::in);
	if(!stream.is_open())
	{
		Log::Send(Log::Error, "Cannot find 'isph.cta'");
		return "";
	}

	struct tar_header {
		char name[100];
		char _unused[24];
		char size[12];
		char _padding[376];
	} header;

	// find the file in archive and read it
	while(!stream.eof())
	{
		stream.read((char*)&header, 512);
		if(header.name[0]==0) 
		{
			text.clear();
			break;
		}

		int fileLength;
		sscanf(header.size, "%o", &fileLength);
		int blockCount = (fileLength + 511) / 512;
		text.resize(blockCount*512);
		stream.read(&text[0], blockCount*512);
		text.resize(fileLength);

		if(filename == header.name)	break;
	}
	
	if(text.empty())
		Log::Send(Log::Error, "Cannot find '" + filename + "' in 'isph.cta'");

	stream.close();
	return text;
}

Vec<2,unsigned int>* Utils::CreateRandomSequence(unsigned int sequenceLength)
{
	Vec<2,unsigned int>* hostHashes = new Vec<2,unsigned int>[sequenceLength];
    srand((unsigned int)time(NULL));
	for ( unsigned int n=0; n < sequenceLength; n++ ) 
	{
        Vec<2,unsigned int> element(n,rand());
        hostHashes[n] = element;
	}
    return hostHashes; 
}

bool Utils::IsPowerOf2( unsigned int num )
{
	return ((num - 1) & num) == 0;
}

unsigned int Utils::FactorRadix2( unsigned int num )
{
	while((num & 1) == 0)
		num >>= 1;
	return num;
}
