#include "vtkwriter.h"
#include "simulation.h"
#include "log.h"
#include <sstream>

using namespace isph;

template<int dim, typename typ>
VtkWriter<dim, typ>::VtkWriter(Simulation<dim,typ>* simulation) :
	Writer<dim, typ>::Writer(simulation),
	fileIndex(0)
{

}

template<int dim, typename typ>
VtkWriter<dim, typ>::~VtkWriter()
{

}

template<int dim, typename typ>
bool VtkWriter<dim, typ>::Prepare()
{
	// semantic names -> real atribute buffers
	for (std::list<std::string>::const_iterator iter = attributeNameList.begin(); iter != attributeNameList.end(); ++iter)
	{
		ParticleAttributeBuffer* att = this->sim->ParticleAttribute(*iter);
		if(att)
			attributeList.push_back(att);
	}

	return true;
}

template<int dim, typename typ>
bool VtkWriter<dim, typ>::Finish()
{
	return true;
}

template<int dim, typename typ>
void VtkWriter<dim, typ>::AddAttribute(const std::string& attName)
{
	attributeNameList.push_back(attName);
}

template<int dim, typename typ>
void VtkWriter<dim, typ>::WriteData()
{
	// since VTK file format is for only one time step, ignore standard header/footer procedure
	// on every step create new file, and write everything at once
	std::stringstream ss;
	ss << this->path << "_" << fileIndex << ".vtk";
	fileIndex++;
	std::string curPath;
	ss >> curPath;

	stream.open(curPath.c_str());

	if(!stream.is_open())
	{
		Log::Send(Log::Error, "Couln't open new VTK export file: "+ curPath);
		return;
	}

	stream.setf(std::ios::scientific);
    
	// write header
    WriteHeader();

	// write positions
    WritePositions();

	// write data
	for (std::list<ParticleAttributeBuffer*>::iterator iter = attributeList.begin(); iter != attributeList.end(); ++iter)
	{
		if (iter == attributeList.begin())
        	stream << "POINT_DATA " << this->sim->ParticleCount()  << std::endl;

		if ((*iter)->DataType() == this->sim->ScalarDataType()) 
		{
            WriteScalarField(*iter);
		}
		else if ((*iter)->DataType() == this->sim->VectorDataType()) 
		{
			WriteVectorField(*iter);
		}
	}

	// close file
	stream << std::endl;
	if(stream.is_open())
		stream.close();
}

template<int dim, typename typ>
void VtkWriter<dim, typ>::WriteHeader()
{
	// write header
	stream << "# vtk DataFile Version 2.0" << std::endl;
	stream << "t = " << this->sim->Time() << " s" << std::endl;
	stream << "ASCII" << std::endl;
	stream << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

template<int dim, typename typ>
void VtkWriter<dim, typ>::WritePositions()
{
	ParticleAttributeBuffer* att = this->sim->ParticleAttribute("POSITIONS");
	att->Download();

	// write positions
	stream << "POINTS " << this->sim->ParticleCount();
	if(this->sim->ScalarPrecision() == 32)
		stream << " float" << std::endl;
	else
		stream << " double" << std::endl;

	for (unsigned int i=0; i < this->sim->ParticleCount(); i++)
	{
		Vec<dim,typ> pos = *(Vec<dim,typ>*)att->Get(i);
		stream << pos.x << " " << pos.y << " ";
		if(dim==3)
			stream << pos[2] << std::endl;
		else
			stream << (typ)0 << std::endl;
	}
    stream << std::endl;
}

template<int dim, typename typ>
void VtkWriter<dim, typ>::WriteScalarField(ParticleAttributeBuffer* att)
{
	att->Download();
		
	stream << "SCALARS " << att->Name() << " ";
	if(this->sim->ScalarPrecision() == 32)
		stream << "float" << std::endl;
	else
		stream << "double" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i=0; i < this->sim->ParticleCount(); i++)
	{
		stream <<  *(typ*)att->Get(i) << std::endl;
	}
}

template<int dim, typename typ>
void VtkWriter<dim, typ>::WriteVectorField(ParticleAttributeBuffer* att)
{
	att->Download();

	stream << "VECTORS " << att->Name() << " ";
	if(this->sim->ScalarPrecision() == 32)
		stream << "float" << std::endl;
	else
		stream << "double" << std::endl;
	
	for (unsigned int i=0; i < this->sim->ParticleCount(); i++)
 	{
	    Vec<dim,typ> v = *(Vec<dim,typ>*)att->Get(i);
	    stream << v.x << " " << v.y << " ";
		if(dim==3)
		  stream << v[2] << std::endl;
		 else
		  stream << (typ)0 << std::endl;
	}
}


//////////////////////////////////////////////////////////////////////////

template class VtkWriter<2,float>;
