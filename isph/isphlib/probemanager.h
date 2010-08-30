#ifndef PROBEMANAGER_WRITER_H
#define PROBEMANAGER_WRITER_H

#include "particle.h"
#include <fstream>
#include <string>
#include <list>
#include <vector>

namespace isph {

	template<int dim, typename typ>	class Simulation;


/*!
 *	\class	ProbeManager
 *	\brief	Class to manage probe data.
 */
template<int dim, typename typ>
class ProbeManager
{
public:

	ProbeManager(Simulation<dim,typ>* simulation);

	~ProbeManager();


	/*!
	 *	\brief	Initializes Probes buffer and file
	 */
	virtual bool Prepare();

	/*!
	 *	\brief	Closes probes file
	 */
	virtual bool Finish();

	/*!
	 *	\brief	Initializes Probes devices variable and kernels
	 */
	void InitKernels();
     
	/*!
	 *	\brief	Enqueues kernels to average required particle attributes at probes location
	 *          every n times steps as specified by sampling frequency.
	 */
	void ReadProbes(int timeStepCount, double timeOverall);

	/*!
	 *	\brief	Set the frequency of sampling in terns of time steps.
	 */
	void SetSamplingFrequency(unsigned int freq) {samplingFrequency = freq;}

	/*!
	 *	\brief	Get the frequency of sampling.
	 */
	inline unsigned int GetSamplingFrequency() { return samplingFrequency; }

	/*!
	 *	\brief	Set the output file path.
	 */
	void SetOutput(const std::string& outputPath)
	{
		path = outputPath;
	}

	/*!
	 *	\brief	Add a new probe at specified location.
	 */
	void AddProbe(const Vec<dim,typ> location);

	/*!
	 *	\brief	Add probes along a line defined by two points with specified spacing
	 *          the first probe will be located on the start point the next one 
	 *          at spacing distance from the first along the line.
	 */
	void AddProbesString(const Vec<dim,typ> start, const Vec<dim,typ> end, typ spacing);

	/*!
	 *	\brief	Add an attribute with a given name.
	 */
	void AddAttribute( const std::string& attName );

	/*!
	 *	\brief	Get the total number of probes.
	 */
	inline unsigned int GetProbesCount(){ return locationList.size(); }

	/*!
	 *	\brief	Get the total number of attributes.
	 */
	inline unsigned int GetAttributesCount(){ return attributeList.size(); }

	/*!
	 *	\brief	Get the size of the probes buffer.
	 */
	inline unsigned int GetBufferSize(){ return bufferSize; }

	/*!
	 *	\brief	A reference to the host buffer.
	 */
	inline typ* GetProbesData() { return data; }

	/*!
	 *	\brief	A reference to the array representation of locations.
	 */
	inline Vec<dim,typ>* GetProbesLocation() {return locations; }

	/*!
	 *	\brief	Writes the content of the buffer to the output.
	 */
	void WriteBuffer();

	/*!
	 *	\brief	Increase the number of buffered steps.
	 */
	inline void RecordSamplingTime( double time ) { times[recordedSteps++] = time; }

	inline unsigned int RecordedSteps() { return recordedSteps; }

    inline unsigned int BufferingSteps() { return bufferingSteps; }

	/*!
	 *	\brief	Increase the number of buffered steps.
	 */
	inline bool BufferFull() { return ( recordedSteps >= bufferingSteps); };

private:
     
	/*!
	 *	\brief	Writes probes file header lines.
	 */
	void WriteHeader();


	Simulation<dim,typ>* sim;
	unsigned int samplingFrequency; // Probes sampling frequency
	unsigned int totalScalarValues;
	unsigned int singleBufferSize;
	unsigned int bufferingSteps;
    unsigned int bufferSize;    
	unsigned int recordedSteps;
    unsigned int recordedValues[2];

	typ* data;
	Vec<dim,typ>* locations;
    double* times;

	std::vector< Vec<dim,typ> > locationList;
	std::list<std::string> attributeNameList;
	std::list<ParticleAttributeBuffer*> attributeList;
    bool initialized;
	bool headerWritten;
	std::string path;
	std::ofstream stream;

};


} // namespace isph

#endif
