#ifndef ISPH_ABSTRACT_INTEGRATOR_H
#define ISPH_ABSTRACT_INTEGRATOR_H


namespace isph {

	template<int dim, typename typ>	class Simulation;


/*!
 *	\class	AbstractIntegrator
 *	\brief	Base class used to implement time integrators.
 */
template<int dim, typename typ>
class AbstractIntegrator
{
public:

	AbstractIntegrator(Simulation<dim,typ>* simulation) : sim(simulation),doRegrid(true) 
	{

	}

	AbstractIntegrator(Simulation<dim,typ>* simulation, bool regrid) : sim(simulation),doRegrid(regrid) 
	{

	}

	virtual ~AbstractIntegrator() 
	{

	}

	/*!
	 *	\brief	Integrate a signle time step.
	 *
	 *	Override the function to implement concrete class.
	 */
	virtual bool Integrate() { return true; }


	/*!
	 *	\brief	Set option to run grid between partial time steps.
	 */
	inline void SetRegrid(bool regrid) {doRegrid = regrid;}
	 
//protected:
// protected should work, but does not. WHY?
public:
	Simulation<dim,typ> *sim;
    bool doRegrid;
};


} // namespace isph

#endif
