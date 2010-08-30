#ifndef ISPH_STDWRITER_H
#define ISPH_STDWRITER_H

#include "writer.h"
#include <fstream>

namespace isph {


/*!
 *	\class	StdWriter
 *	\brief	Abstract class for exporting simulated data to a file with STL IO.
 */
template<int dim, typename typ>
class StdWriter : public Writer<dim,typ>
{
public:

	using Writer<dim,typ>::path;
	
	StdWriter(Simulation<dim,typ>* simulation) : Writer<dim,typ>::Writer(simulation)
	{
		binary = false;
	}

	virtual ~StdWriter() 
	{
		Finish();
	}

	virtual bool Prepare()
	{
		if (path.empty())
			return false;
		
		std::ios_base::open_mode openMode = std::ios::out | std::ios::trunc;
		if(binary)
			openMode |= std::ios::binary;

		stream.open(path.c_str());
		
		return stream.is_open();
	}

	virtual bool Finish()
	{
		if(stream.is_open())
			stream.close();
		return true;
	}

	virtual void WriteData() = 0;

protected:

	bool binary;
	std::ofstream stream;

};


} // namespace isph

#endif
