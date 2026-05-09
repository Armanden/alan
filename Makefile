CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

all: alan

alan: main.c
	$(CC) $(CFLAGS) -o alan main.c $(LDFLAGS)

clean:
	rm -f alan

run: alan
	./alan

.PHONY: all clean run
