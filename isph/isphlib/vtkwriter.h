#ifndef ISPH_VTKWRITER_H
#define ISPH_VTKWRITER_H

#include "writer.h"
#include "particle.h"
#include <list>
#include <fstream>

namespace isph {

	/*!
	 *	\class	VtkWriter
	 *	\brief	Exporting simulated data to a VTK legacy file-format.
	 */
	template<int dim, typename typ>
	class VtkWriter : public Writer<dim,typ>
	{
	public:

		VtkWriter(Simulation<dim,typ>* simulation);

		virtual ~VtkWriter();

		virtual bool Prepare();

		virtual bool Finish();

		virtual void WriteData();

		virtual void AddAttribute(const std::string& attName);

		virtual inline unsigned int FileIndex(){ return fileIndex; }

	protected:

		virtual void WriteHeader();

		virtual void WritePositions();

		virtual void WriteScalarField(ParticleAttributeBuffer* att);

		virtual void WriteVectorField(ParticleAttributeBuffer* att);


		unsigned int fileIndex;
		std::ofstream stream;
		std::list<std::string> attributeNameList;
		std::list<ParticleAttributeBuffer*> attributeList;
    
	};

} // namespace isph

#endif
