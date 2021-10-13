BLIS_PATH := ../../../..

-include $(BLIS_PATH)/config.mk

INC_PATH  = $(BLIS_PATH)/include/$(CONFIG_NAME)
LIBBLIS_L = $(BLIS_PATH)/lib/$(CONFIG_NAME)/libblis.a

LINKER := $(CC)
CFLAGS := -I $(INC_PATH)
LDFLAGS := -lpthread -lm

OBJS      := $(patsubst %.c,%.o, $(wildcard *.c))
PERF_OBJS := performance.o
COR_OBJS  := correctness.o

all: performance correctness

$(OBJS): %.o: %.c %.h common.h ../../bli_sandbox.h $(INC_PATH)/blis.h
	$(CC) $(CFLAGS) -c $< -o $@

performance: $(PERF_OBJS)
	$(LINKER) $(PERF_OBJS) $(LIBBLIS_L) -o ./gather_perf.x $(LDFLAGS) 

correctness: $(COR_OBJS)
	$(LINKER) $(COR_OBJS) $(LIBBLIS_L) -o ./test_correctness.x $(LDFLAGS) 

csv_clean:
	rm -rf *.csv

clean: 
	rm -rf *.x *.o
