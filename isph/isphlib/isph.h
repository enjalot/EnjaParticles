#ifndef ISPH_LIB_H
#define ISPH_LIB_H

// OpenCL framework
#include "clsystem.h"
#include "clplatform.h"
#include "cllink.h"
#include "cldevice.h"
#include "clprogram.h"
#include "clsubprogram.h"
#include "clvariable.h"

// utilities
#include "vec.h"
#include "log.h"
#include "utils.h"

// simulation
#include "wcsphsimulation.h"
#include "pcisphsimulation.h"

//integrator
#include "pcintegrator.h"

// exporters
#include "vtkwriter.h"

#endif
