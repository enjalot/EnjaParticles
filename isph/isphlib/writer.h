#ifndef ISPH_WRITER_H
#define ISPH_WRITER_H

#include <string>


namespace isph {

	template<int dim, typename typ>	class Simulation;


/*!
 *	\class	Writer
 *	\brief	Abstract class for exporting simulated data.
 */
template<int dim, typename typ>
class Writer
{
public:

	Writer(Simulation<dim,typ>* simulation) : sim(simulation) 
	{

	}

	virtual ~Writer() 
	{
		Finish();
	}

	/*!
	 *	\brief	Set the output file path.
	 */
	void SetOutput(const std::string& outputPath)
	{
		path = outputPath;
	}

	/*!
	 *	\brief	Prepare the file for writing.
	 *
	 *	Override the function to create the output file and prepare it for writing.
	 */
	virtual bool Prepare() { return true; }

	/*!
	 *	\brief	Finish writing the file.
	 *
	 *	Override the function to close the output file after writing. It can also be used for writing recorded
	 *	particle properties. This means that if you do not want time-based, but particle-based exported
	 *	file for example, with WriteData you can track particle, and with this function you can
	 *	write all recorded data.
	 */
	virtual bool Finish() { return true; }

	/*!
	 *	\brief	Write simulated data-set for current time.
	 */
	virtual void WriteData() = 0;

	std::string path;
	Simulation<dim,typ> *sim;

};


} // namespace isph

#endif
