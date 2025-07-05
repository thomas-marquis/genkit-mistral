package mistral

import (
	"log"
	"os"
)

var (
	logger = log.New(os.Stdout, "genkit-mistral: ", log.LstdFlags|log.Lshortfile)
)
