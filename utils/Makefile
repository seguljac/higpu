
CC=g++

PROGS=extr_stdc_hdrs

# We must compile a 32-bit binary.
extr_stdc_hdrs: extr_stdc_hdrs.cpp
	$(CC) -m32 -O3 -o $@ $<

clean:
	rm -rf $(PROGS)
