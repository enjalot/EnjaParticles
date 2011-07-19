###
#   Author: Gaspard Petit,
#           Etienne Clement,
#           Christopher Jensen
#
#	Last Modification:
#           March 14, 2007
#
###

CC = gcc
CXX = g++
LINKCC = $(CXX)
LIBTOOL = libtool

OS_NAME := $(shell uname -s)


CPPFLAGS := -Weffc++
CFLAGS := -Wall -D$(OS_NAME)
LDFLAGS :=
INCLUDE :=

SRCDIR = .
DEPSDIR = .dep
OBJSDIR = obj
EXEDIR = .

# DEFAULT MODE
ifeq ($(MODE),)
MODE=release
endif

# CONFIGURATIONS
ifeq ($(MODE), debug)
CFLAGS += -g -ggdb3
endif

ifeq ($(MODE), prof)
CFLAGS += -g -ggdb3 -pg
LDFLAGS += -pg
endif

ifeq ($(MODE), release)
CFLAGS += -O3
endif

# COLORGCC
PATH := $(addprefix .:, $(PATH))
HASCOLOR = $(shell if test `which colorgcc`; then echo true; else echo false; fi)
ifneq ($(HASCOLOR),true)
HASCOLOR = $(shell if test -e colorgcc; then echo true; else echo false; fi)
endif

ifeq ($(HASCOLOR),true)
ifneq ($(COLOR), false)
CXX := $(addprefix colorgcc , $(CXX))
CC := $(addprefix colorgcc , $(CC)) 
endif
endif



MAKEFILEDEP := Makefile

ifeq ($(shell if test -e $(DEPSDIR); then echo true; else echo false; fi),false)
ALL_TARGET := $(DEPSDIR) $(ALL_TARGET)
endif

ifeq ($(shell if test -e $(OBJSDIR); then echo true; else echo false; fi),false)
ALL_TARGET := $(OBJSDIR) $(ALL_TARGET)
endif


HASMAKEFILEDOTSPEC = $(shell if test -e Makefile.spec; then echo true; else echo false; fi)
ifeq ($(HASMAKEFILEDOTSPEC),true)
include Makefile.spec
MAKEFILEDEP += Makefile.spec
else
ALL_TARGET += info
endif

ifneq ($(ARCHIVE),)
ARCHIVE := $(ARCHIVE).a
endif

ifneq ($(DYNLIB),)
ifeq ($(OS_NAME),Darwin)
DYNLIB := $(DYNLIB).dylib
else
DYNLIB := $(DYNLIB).so
endif
endif


ifneq ($(SUBDIR),)
ALL_TARGET := subdir $(ALL_TARGET)
endif

ifneq ($(ARCHIVE),)
ALL_TARGET += archive
endif

ifneq ($(EXECUTABLE),)
ALL_TARGET += executable
endif

ifneq ($(DYNLIB),)
ALL_TARGET += dynlib
endif

ifeq ($(OS_NAME),Darwin)
DYNLIBFLAG = -arch ppc i386 -Wl,-single_module -dynamiclib -compatibility_version 1 -current_version 1 -install_name /usr/local/lib/$(DYNLIB)
else
flags not set for linux...
endif


SRCS := $(wildcard $(SRCDIR)/*.cpp $(SRCDIR)/*.c $(SRCDIR)/*.m $(SRCDIR)/*.mm $(SRCDIR)/*.h)

DEPS += $(patsubst $(SRCDIR)/%.m,$(DEPSDIR)/%.d,$(wildcard $(SRCDIR)/*.m))
DEPS += $(patsubst $(SRCDIR)/%.c,$(DEPSDIR)/%.d,$(wildcard $(SRCDIR)/*.c))
DEPS += $(patsubst $(SRCDIR)/%.mm,$(DEPSDIR)/%.d,$(wildcard $(SRCDIR)/*.mm))
DEPS += $(patsubst $(SRCDIR)/%.cpp,$(DEPSDIR)/%.d,$(wildcard $(SRCDIR)/*.cpp))

OBJS += $(patsubst $(SRCDIR)/%.m,$(OBJSDIR)/%.o,$(wildcard $(SRCDIR)/*.m))
OBJS += $(patsubst $(SRCDIR)/%.c,$(OBJSDIR)/%.o,$(wildcard $(SRCDIR)/*.c))
OBJS += $(patsubst $(SRCDIR)/%.mm,$(OBJSDIR)/%.o,$(wildcard $(SRCDIR)/*.mm))
OBJS += $(patsubst $(SRCDIR)/%.cpp,$(OBJSDIR)/%.o,$(wildcard $(SRCDIR)/*.cpp))


.PHONY: all clean explain info executable clean_executable objects cleanobj dependency cleandep subdir $(SUBDIR)

all:  $(ALL_TARGET)

clean: clean_executable
executable: $(EXECUTABLE)
archive: $(ARCHIVE)
subdir: $(SUBDIR)
	
$(SUBDIR):
	@echo
	@echo "Making sure \"$@\" is up to date..."
	make -C $@
	@echo
	
$(EXECUTABLE): $(OBJS) $(LINKWITH)
	@echo "##################[$(notdir $@)]################"
	$(LINKCC) -o $@ $(OBJS) $(LDFLAGS) $(LINKWITH)
	@echo

$(ARCHIVE):	$(OBJS)
	@echo "##################[$(notdir $@)]################"
	ar rc "$@" $(OBJS)
	ranlib "$@"
	@echo

$(DYNLIB): $(OBJS) $(LINKWITH)
	@echo "##################[$(notdir $@)]################"
	$(LINKCC) -o $@ $(OBJS) $(LDFLAGS) $(LINKWITH) $(DYNLIBFLAG)
	@echo

clean_executable: cleanobj
	rm -f $(EXECUTABLE)

objects: $(OBJS)

dynlib: $(DYNLIB)

cleanobj: cleandep
	rm -rf $(OBJSDIR)/*.o
	-rmdir $(OBJSDIR)

dependency: $(DEPS)
cleandep:
	rm -rf $(DEPSDIR)/*.d
	-rmdir $(DEPSDIR)

$(OBJSDIR):
	mkdir $(OBJSDIR)

$(DEPSDIR):
	mkdir $(DEPSDIR)
	
$(OBJSDIR)/%.o: $(SRCDIR)/%.cpp $(DEPSDIR)/%.d
	@echo
	@echo "=====================[$(notdir $@)]====================="
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(addprefix -I, $(INCLUDE)) -c -o $@ $(SRCDIR)/$(patsubst %.d,%.cpp,$(notdir $<))
	@echo

$(OBJSDIR)/%.o: $(SRCDIR)/%.c $(DEPSDIR)/%.d
	@echo
	@echo "=====================[$(notdir $@)]====================="
	$(CC) $(CFLAGS) $(addprefix -I, $(INCLUDE)) -c -o $@ $(SRCDIR)/$(patsubst %.d,%.c,$(notdir $<))
	@echo

$(OBJSDIR)/%.o: $(SRCDIR)/%.m $(DEPSDIR)/%.d
	@echo
	@echo "=====================[$(notdir $@)]====================="
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(addprefix -I, $(INCLUDE)) -c -o $@ $(SRCDIR)/$(patsubst %.d,%.m,$(notdir $<))
	@echo

$(OBJSDIR)/%.o: $(SRCDIR)/%.mm $(DEPSDIR)/%.d
	@echo
	@echo "=====================[$(notdir $@)]====================="
	$(CC) $(CFLAGS) $(addprefix -I, $(INCLUDE)) -c -o $@ $(SRCDIR)/$(patsubst %.d,%.mm,$(notdir $<))
	@echo


$(DEPSDIR)/%.d: $(SRCDIR)/%.cpp $(MAKEFILEDEP)
	@echo
	@echo "--------------------[$(notdir $@)]-----------------"

	$(CXX) -M $(CFLAGS) $(CPPFLAGS) $(addprefix -I, $(INCLUDE)) $< | sed "s|$(patsubst %.cpp,%.o,$<):|$(DEPSDIR)/$(patsubst %.cpp,%.d,$<):|" > $@.tmp
	@cat $@.tmp > $@
	@rm "$@.tmp"
	@echo

$(DEPSDIR)/%.d: $(SRCDIR)/%.c $(MAKEFILEDEP)
	@echo
	@echo "--------------------[$(notdir $@)]-----------------"
	$(CC) -M $(CFLAGS) $(CPPFLAGS) $(addprefix -I, $(INCLUDE)) $< | sed "s|$(patsubst %.c,%.o,$<):|$(DEPSDIR)/$(patsubst %.c,%.d,$<):|" > $@.tmp
	@cat $@.tmp > $@
	@rm "$@.tmp"
	@echo

$(DEPSDIR)/%.d: $(SRCDIR)/%.mm $(MAKEFILEDEP)
	@echo
	@echo "--------------------[$(notdir $@)]-----------------"
	$(CC) -M $(CFLAGS) $(CPPFLAGS) $(addprefix -I, $(INCLUDE)) $< | sed "s|$(patsubst %.mm,%.o,$<):|$(DEPSDIR)/$(patsubst %.mm,%.d,$<):|" > $@.tmp
	@cat $@.tmp > $@
	@rm "$@.tmp"
	@echo

$(DEPSDIR)/%.d: $(SRCDIR)/%.m $(MAKEFILEDEP)
	@echo
	@echo "--------------------[$(notdir $@)]-----------------"
	$(CC) -M $(CFLAGS) $(CPPFLAGS) $(addprefix -I, $(INCLUDE)) $< | sed "s|$(patsubst %.m,%.o,$<):|$(DEPSDIR)/$(patsubst %.m,%.d,$<):|" > $@.tmp
	@cat $@.tmp > $@
	@rm "$@.tmp"
	@echo


explain:
	@echo "================================================================================"
	@echo "The following information represents your directory:"

	@echo
ifeq ($(HASMAKEFILEDOTSPEC),true)
	@echo "Makefile.spec: installed"
else
	@echo "Makefile.spec: not found"
endif

ifeq ($(HASCOLOR),true)
	@echo "colorgcc:      installed"
else
	@echo "colorgcc:      not found"
endif
	@echo

	@echo "Source files:"
	@for name in $(SRCS); do echo "    $$name"; done
	@echo

	@echo "Object files:"
	@for name in $(OBJS); do echo "    $$name"; done
	@echo

	@echo "Dependency files:"
	@for name in $(DEPS); do echo "    $$name"; done
	@echo

	@echo "Archive:"
	@for name in $(ARCHIVE); do echo "    $$name"; done
	@echo

	@echo "Executable:"
	@for name in $(EXECUTABLE); do echo "    $$name"; done
	@echo

	@echo "Dependency Projects:"
	@for name in $(SUBDIR); do echo "    $$name"; done
	@echo

ifeq ($(HASMAKEFILEDOTSPEC),true)
Makefile.spec:
	@echo
	@echo "Makefile.spec already exists - not overwriting, please remove"
	@echo "the file first..."
	@echo
else
Makefile.spec:
	@echo "# Default Makefile.spec" > Makefile.spec
	@echo >> Makefile.spec
	@echo "#Additional include paths" >> Makefile.spec
	@echo "INCLUDE +=" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Additional C flags" >> Makefile.spec
	@echo "CFLAGS +=" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Additional C++ flags" >> Makefile.spec
	@echo "CPPFLAGS +=" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#List of libraries to link with" >> Makefile.spec
	@echo "LINKWITH +=" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Additional Linker flags" >> Makefile.spec
	@echo "LDFLAGS +=" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Name of the executable to create" >> Makefile.spec
	@echo "EXECUTABLE =" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Name of the static library to create" >> Makefile.spec
	@echo "ARCHIVE =" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Name of the dynamic library to create" >> Makefile.spec
	@echo "DYNLIB =" >> Makefile.spec
	@echo >> Makefile.spec
	@echo "#Other directories to update before compiling" >> Makefile.spec
	@echo "SUBDIR =" >> Makefile.spec
	@echo
	@echo "Makefile.spec file created"
	@echo
endif

colorgcc:
ifeq ($(shell if test -e colorgcc; then echo true; else echo false; fi),true)
	@echo "colorgcc already exists in current directory - please remove it first..."
else
	@echo 'echo -e "begin-base64 644 colorgcc.gz\n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\n====" | uudecode -p | gunzip' | /bin/bash > colorgcc
	@chmod +x colorgcc
endif

info:
	@echo
	@echo "You should to write a Makefile.spec file containing your target"
	@echo "as well as any special configuration.  For instance:"
	@echo
	@echo "INCLUDE += <any other include path>"
	@echo "CFLAGS += <any other compiler flag>"
	@echo "LDFLAGS += <any other linker flag>"
	@echo "LINKWITH += <list of library to link with>"
	@echo "EXECUTABLE = <name of the executable to create>"
	@echo "ARCHIVE = <name of the static library to create>"
	@echo "DYNLIB = <name of the dynamic library to create>"
	@echo "SUBDIR += <directory of dependency projects>"
	@echo
	@echo "Object files are created in a \"obj\" directory and dependency"
	@echo "files are hidden in a \".dep\" directory.  If you have problems"
	@echo "after adding or removing a source file, you should try deleting"
	@echo "the dependency file (or the whole dependency directory if you're"
	@echo "lazy)."
	@echo
	@echo "If you want to know what's going to happen, try:"
	@echo "    make explain"
	@echo
	@echo "If you'd like a default \"Makefile.spec\" to be conveniently"
	@echo "created for you, type"
	@echo "    make Makefile.spec"
	@echo
	@echo "Finally, if you don't care about all this and just want to compile"
	@echo "your files into .o files, type:"
	@echo "    make objects"
	@echo



ifeq ($(shell if test -d $(DEPSDIR); then echo true; else echo false; fi),true)
ifneq ($(DEPS),)
-include $(DEPS)
endif
endif
