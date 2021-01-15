all:
	@./build.sh
clean:
	@rm -f facemask
install: all
	@cp facemask /usr/local/bin
uninstall:
	@rm -f /usr/local/bin/facemask
package:
	@NOCOPY=1 ./build.sh package