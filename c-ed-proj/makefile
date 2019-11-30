CC=gcc
LIBDIR=/usr/lib
INCLUDEDIR=/usr/include
CFLAGS=-Iinclude -I$(INCLUDEDIR) -I$(LIBDIR) -lm -lblas -llapack
LDFLAGS=
SRCDIR=src
SOURCE_FNAMES=main.c states.c ham.c transform.c geometry.c print_array.c utility.c
SOURCES=$(addprefix $(SRCDIR)/,$(SOURCE_FNAMES))
#SOURCES=loop-basis.c states.c ham.c transform.c geometry.c
OBJDIR=obj
OBJECTS=$(addprefix $(OBJDIR)/,$(SOURCE_FNAMES:.c=.o))
EXEDIR=bin
EXECUTABLE=$(addprefix $(EXEDIR)/,loop-basis.exe)

#TODO: need to properly include headers as dependencies...

all: $(OBJECTS) | $(EXEDIR)
	$(CC) $(OBJECTS) $(CFLAGS) -o $(EXECUTABLE)

obj/main.o: $(SOURCES) | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

obj/states.o: src/states.c | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

obj/ham.o: src/ham.c include/states.h | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

obj/transform.o: src/transform.c include/states.h include/utility.h| $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

obj/geometry.o:  src/geometry.c | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

obj/print_array.o: src/print_array.c | $(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@

obj/utility.o: src/utility.c |$(OBJDIR)
	$(CC) -c $(CFLAGS) $< -o $@


#$(OBJECTS): $(SOURCES) | $(OBJDIR)
#$(CC) -c $(CFLAGS) $(SOURCES) -o $@



$(OBJDIR):
	mkdir $(OBJDIR)

$(EXEDIR):
	mkdir $(EXEDIR)

clean:
	rm $(EXECUTABLE) $(OBJECTS)


#general syntax for a rule is...
#target: prequisites
#	rule


#
#$< holds name of first prerequisite
#$@ holds name of target
