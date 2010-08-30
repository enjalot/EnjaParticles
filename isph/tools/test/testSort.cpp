#include <gpubitonicsort.h>
#include <isph.h>
#include <time.h>
#include <utils.h>
#include <iostream>

using namespace std;
using namespace isph;

int main(int argc, char *argv[])
{
	Log::SetOutput("test_log.txt"); // output everything to log file

	// print OpenCL supported devices

	CLSystem* sys = CLSystem::Instance();

	cout << "Platforms: " << sys->PlatformCount() << endl;

	for(unsigned int i=0; i<sys->PlatformCount(); i++)
	{
		CLPlatform* pl = sys->Platform(i);
		cout << endl << "Platform " << i << ": " << pl->Name() << endl;
		cout << "Device count: " << pl->DeviceCount() << endl;

		for (unsigned int j=0; j<pl->DeviceCount(); j++)
		{
			CLDevice* dev = pl->Device(j);
			cout << endl << "  Device " << j << ": " << dev->Name() << endl;
			cout << "  Cores: " << dev->Cores() << endl;
			cout << "  Max frequency: " << dev->MaxFrequency() << " MHz" << endl;
			cout << "  Global memory: " << dev->GlobalMemorySize() << " bytes" << endl;
			cout << "  Local memory: " << dev->LocalMemorySize() << " bytes" << endl;
			cout << "  Double floating precision: " << dev->DoublePrecision() << endl;
			cout << "  Half floating precision: " << dev->HalfPrecision() << endl;
			cout << "  Atomic functions: " << dev->Atomics() << endl;
		}
	}

	CLProgram* program = new CLProgram();
    program->SetLink(new CLLink(sys->FirstPlatform()->Device(0)));
    
    bool doLog= false;
	unsigned int maxLength =500000;
    unsigned int cell_hash[500000] ;
    unsigned int particle_id[500000] ;
	for (unsigned int sLength=400000;sLength<maxLength;sLength++) 
	{
	// Create sequence
	unsigned long sequenceLength = sLength;//0X1<<4;
	Vec<2,unsigned int>* hostHashes =  Utils::CreateRandomSequence(sequenceLength);
	for ( unsigned int n=0; n < sequenceLength; n++ ) 
	{
       particle_id[n] = hostHashes[n].x;
       cell_hash[n] = hostHashes[n].y;
	}

	if (doLog) 
	{
	  for ( unsigned int n=0; n < sequenceLength; n++ ) 
	  {

		cout<< "Id: "<< hostHashes[n].x  << " hash:"  << hostHashes[n].y <<endl;
	  }
	}
    cout << "Created " << sequenceLength << " elements sequence." << endl;
    int error;
	CLVariable* vhashes = new CLVariable(program, "CELLS_HASH");
	vhashes ->SetSpace(sequenceLength , 4);
	vhashes ->SetAccess(true, false);
	error = vhashes ->WriteFrom(cell_hash);

	CLVariable* vids= new CLVariable(program, "HASHES_PARTICLE");
	vids->SetSpace(sequenceLength , 4);
	vids->SetAccess(true, false);
	error = vids->WriteFrom(particle_id);

	// Initialize sorter   
	//CpuBitonicSort* s =  new CpuBitonicSort( sequenceLength);//TODO calculate array length  
	GpuBitonicSort* s = new GpuBitonicSort( program->Link()->Context(), program->Link()->Queue(0),sequenceLength);
	cout << "Bitonic Sort Start" << endl;
	//s->sort(hostHashes);
	s->sort(program->Variable("CELLS_HASH")->Buffer(0), program->Variable("HASHES_PARTICLE")->Buffer(0));
	cout << "Bitonic Sort End" << endl;
	cout << "Checking sorted sequence...." << endl;
    vids->ReadTo(particle_id);
	vhashes ->ReadTo(cell_hash);

	unsigned int maxKey =0; 
	for ( unsigned int n=0; n < sequenceLength; n++ ) 
	{
        hostHashes[n].x = particle_id[n];
        hostHashes[n].y = cell_hash[n];
	}
	for ( unsigned int n=1; n < sequenceLength; n++ ) 
	{
        if (hostHashes[n].y  < hostHashes[n - 1].y  ) 
 		{  
			cout << "Sorting error - Position n:" << n << " key: " << hostHashes[n].y;
			cout << ", Position n-1:" << (n - 1) << " key: " << hostHashes[n - 1].y << endl;
		    
			exit(1);
		}
		if (hostHashes[n].y  > maxKey) maxKey = hostHashes[n].y;   
		if ( (n<100) )
		{
			cout << "Id: " << hostHashes[n].x  << " hash:"  << hostHashes[n].y << endl;
		}
	}
	cout << "Successful MaxKey: " << maxKey << endl; 
	}
#ifdef WIN32
	cin.get();
#endif

	return 0;
}


Vec<2,unsigned int>* createRandomSequence(unsigned int sequenceLength)
{
	Vec<2,unsigned int>* hostHashes = new Vec<2,unsigned int>[sequenceLength];
    srand(time(NULL));
	for ( unsigned int n=0; n < sequenceLength; n++ ) 
	{
        Vec<2,unsigned int> element(n,rand());
        hostHashes[n] = element;
		cout<< "Id: "<< element.x  << " hash:"  << element.y <<endl;
	}
    return hostHashes; 
}
