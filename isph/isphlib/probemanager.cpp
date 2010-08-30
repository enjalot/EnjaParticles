#include "probemanager.h"
#include "isph.h"

#include <sstream>
#include <iomanip>


using namespace isph;

template<int dim, typename typ>
ProbeManager<dim, typ>::ProbeManager(Simulation<dim,typ>* simulation) : 
sim(simulation), 
samplingFrequency(1),
path("probes.txt"),
initialized(false),
totalScalarValues(0),
headerWritten(false)
{

}

template<int dim, typename typ>
ProbeManager<dim, typ>::~ProbeManager()
{
    delete [] data;
	delete [] locations;
	delete [] times;
} 

template<int dim, typename typ>
bool ProbeManager<dim, typ>:: Prepare()
{
	if (path.empty())
		return false;
	
	std::ios_base::open_mode openMode = std::ios::out | std::ios::trunc;

	stream.open(path.c_str());
		
	if (!stream.is_open())
  	    return false;
   
	WriteHeader();

	return true;
}

template<int dim, typename typ>
bool ProbeManager<dim, typ>:: Finish()
{
	if(stream.is_open())
		stream.close();
	return true;
}


template<int dim, typename typ>
void ProbeManager<dim, typ>::AddAttribute(const std::string& attName)
{
	attributeNameList.push_back(attName);
}

template<int dim, typename typ>
void ProbeManager<dim, typ>::AddProbe(const Vec<dim,typ> location)
{
	if (initialized) return;
	locationList.push_back( location );
	singleBufferSize = totalScalarValues*locationList.size();
}

template<int dim, typename typ>
void ProbeManager<dim, typ>::AddProbesString(const Vec<dim,typ> start, const Vec<dim,typ> end, typ spacing)
{
	if (initialized) return;
    Vec<dim,typ> diff = end - start;
	typ length = diff.length();
	typ cumulate = 0;
	while (cumulate<length) 
	{
	   Vec<dim,typ> location = start + 	cumulate * diff / length; 
       AddProbe(location);
	   cumulate += spacing;
	}
}


template<int dim, typename typ>
void ProbeManager<dim, typ>::WriteHeader()
{
	if(!attributeList.size())
		return;

	// write header
	stream << "# ISPH Probes file Version 1.0" << std::endl;
    stream << "# Monitored attributes " << std::endl;
    stream << "n = " << ProbeManager<dim, typ>::attributeList.size() << std::endl;

	// write data
	for (std::list<ParticleAttributeBuffer*>::const_iterator iter = attributeList.begin(); iter != attributeList.end(); ++iter)
	{
        stream << (*iter)->Name();
		if ((*iter)->DataType() == sim->ScalarDataType()) 
		{
			stream << " components: 1" << std::endl; 
		}
		else
		{
			stream << " components: " << dim << std::endl; 
		}
	}
   
	stream << "# Monitored locations " << std::endl;
    stream << "m = " << ProbeManager<dim, typ>::locationList.size() << std::endl;
    
	typename std::vector< Vec<dim,typ> >::iterator it;
	for (it = locationList.begin(); it != locationList.end(); ++it)
	{

		stream << (*it).x << " " << (*it).y << " ";
		if(dim==3)
			stream << (*it)[2] << std::endl;
		else
			stream << (typ)0 << std::endl;
	}

	stream << "# Sampling frequency" << std::endl;
	stream << "t = " << ProbeManager<dim, typ>::samplingFrequency<< " iteration" << std::endl;
    // Write format decription lines
	// Line 1 locations
    stream << "#" << std::left << std::endl;
	// Line 2 time, attributes
	stream << "#" << std::left << std::setw(15) << "time";

	for (std::list<ParticleAttributeBuffer*>::const_iterator iter = attributeList.begin(); iter != attributeList.end(); ++iter)
	{
		if ((*iter)->DataType() == sim->ScalarDataType()) 
		{
			stream << "|" << std::left << std::setw(16*(locationList.size())) << (*iter)->Name();
		}
		else
		{
			std::string tmp;
			tmp = (*iter)->Name() + "(x)";
	        stream <<  "|" << std::left << std::setw(16*(locationList.size())) << tmp;
			tmp = (*iter)->Name() + "(y)";
	        stream <<  "|" << std::left << std::setw(16*(locationList.size())) << tmp;
		}
	}
	stream << std::endl;
}

template<int dim, typename typ>
void ProbeManager<dim, typ>::WriteBuffer()
{
	if(!attributeList.size())
		return;

   sim->program->Variable("PROBES_BUFFER")->ReadTo(data);
   for (unsigned int h=0; h<recordedSteps; h++ )
   {
	  stream << std::scientific << std::setprecision(6)<< std::left << std::setw(15) << times[h] << " ";  
      for ( unsigned int i=0; i<totalScalarValues; i++ )
	      for ( unsigned int j=0; j<locationList.size(); j++ )
	      {
			  stream << std::scientific << std::setprecision(6)<< std::right<< std::setw(15) << data[h*singleBufferSize +  i*totalScalarValues + j] << " ";
	      }
      stream << std::endl;
   }
   recordedSteps = 0;
}

template<int dim, typename typ>
void ProbeManager<dim, typ>::InitKernels() 
{
	// semantic names -> real atribute buffers
	for (std::list<std::string>::const_iterator iter = attributeNameList.begin(); iter != attributeNameList.end(); ++iter)
	{
		ParticleAttributeBuffer* att = sim->ParticleAttribute(*iter);
		if(!att) continue;

		attributeList.push_back(att);

		// Update total number of scalar values per probe
		if (att->DataType() == sim->ScalarDataType()) 
		{ 
			totalScalarValues++;
		}
		else
		{
			totalScalarValues += dim;
		}
	}

	if(!attributeList.size())
		return;

	singleBufferSize = totalScalarValues * locationList.size();

    bufferingSteps = 10;//32768 / singleBufferSize;
    bufferSize = singleBufferSize * bufferingSteps;
		
	// Allocate host buffers
	data = new typ[bufferSize];
	//realLocationSize
    times = new double[bufferingSteps];
 
	// Copy locations to array
	locations = new Vec<dim,typ>[locationList.size()];
	typename std::vector< Vec<dim,typ> >::iterator it;
	unsigned int cnt;
	cnt = 0;
	for (it = locationList.begin(); it != locationList.end(); ++it)
	{
        locations[cnt] = *(it);
	    cnt++;
 	}

	recordedValues[0] = 0;
	recordedValues[1] = 0;
    recordedSteps = 0;

	// Allocate device variables
    sim->InitSimulationBuffer("PROBES_BUFFER", sim->ScalarDataType(), bufferSize);
	sim->InitSimulationBuffer("PROBES_LOCATION", sim->VectorDataType(), locationList.size());
	sim->program->Variable("PROBES_LOCATION")->WriteFrom(locations);
	sim->program->ConnectSemantic("PROBES_SCALAR", sim->program->Variable("PRESSURES"));
    sim->program->ConnectSemantic("PROBES_VECTOR", sim->program->Variable("VELOCITIES"));

	sim->InitSimulationConstant("PROBES_BUFFERING_STEPS", UintType, &bufferingSteps);
	sim->InitSimulationConstant("PROBES_SINGLE_BUFFER_SIZE", UintType, &singleBufferSize);
	
	cnt = locationList.size();
	sim->InitSimulationConstant("PROBES_COUNT", UintType, &cnt);
    cnt = attributeList.size();
	sim->InitSimulationConstant("PROBES_ATT_COUNT", UintType, &totalScalarValues);

	sim->InitSimulationBuffer("PROBES_RECORDED_VALUES", UintType, 2);
	sim->program->Variable("PROBES_RECORDED_VALUES")->WriteFrom(recordedValues);

	// Load kernels
	sim->LoadSubprogram("read probes scalar", "wcsph/read_probes_scalar.cl");
	sim->LoadSubprogram("read probes vector", "wcsph/read_probes_vector.cl");

    initialized = true;
}

template<int dim, typename typ>
void ProbeManager<dim, typ>::ReadProbes(int timeStepCount, double timeOverall) 
{
	// Initialize the first time
	if (!headerWritten) 
	{
		WriteHeader();
		headerWritten = true;
	}

	if(!attributeList.size())
		return;

	if (timeStepCount%samplingFrequency != 0)
		return;
    
	RecordSamplingTime(timeOverall);

    // Cycle trought all required attributes
	for (std::list<ParticleAttributeBuffer*>::const_iterator iter = attributeList.begin(); iter != attributeList.end(); ++iter)
	{
		if ((*iter)->DataType() == sim->ScalarDataType()) 
		{
			// TODO Force execution on single workgroup do we need this?
     	    sim->program->ConnectSemantic("PROBES_SCALAR", (*iter)->DeviceData());
        	sim->EnqueueSubprogram("read probes scalar",GetProbesCount(), GetProbesCount());
		}
		else
		{
			// Force execution on singe workgroup
     	    sim->program->ConnectSemantic("PROBES_VECTOR", (*iter)->DeviceData());
        	sim->EnqueueSubprogram("read probes vector",GetProbesCount(), GetProbesCount());
		}
	}

	// Dowload data and write to file if buffer full
	if (BufferFull()) WriteBuffer();



}


//////////////////////////////////////////////////////////////////////////

template class ProbeManager<2,float>;
