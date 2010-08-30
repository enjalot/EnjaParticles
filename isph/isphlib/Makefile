.SUFFIXES: .cpp .o

.cpp.o:
	g++ -g -c $<

main: probemanager.o clsubprogram.o gpubitonicsort.o vtkwriter.o clsystem.o cldevice.o cpubitonicsort.o log.o pcisphsimulation.o wcsphsimulation.o diffusionsimulation.o clprogram.o cllink.o geometry.o utils.o stdwriter.o particle.o simulation.o writer.o clplatform.o clvariable.o vec.o
	ar rcs ../lib/libisph.a probemanager.o clsubprogram.o gpubitonicsort.o vtkwriter.o clsystem.o cldevice.o cpubitonicsort.o log.o pcisphsimulation.o wcsphsimulation.o diffusionsimulation.o clprogram.o cllink.o geometry.o utils.o stdwriter.o particle.o simulation.o writer.o clplatform.o clvariable.o vec.o
	tar -cvf ../lib/isph.cta --exclude .svn wcsph pcisph general kernels scene integrators
	ln ../lib/isph.cta ../bin/isph.cta

clean:
	rm -f *.o *~ ../bin/isph.cta
