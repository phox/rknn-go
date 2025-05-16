// Command line tool for RKNN model conversion
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/phox/rknn-go/tools/convert"
)

func main() {
	// Configure logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// Check if any command is provided
	if len(os.Args) < 2 {
		convert.Usage()
		os.Exit(1)
	}

	// Parse command
	switch os.Args[1] {
	case "convert":
		// Run the convert command
		if err := convert.RunConvertCommand(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	case "help":
		convert.Usage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", os.Args[1])
		convert.Usage()
		os.Exit(1)
	}
}
