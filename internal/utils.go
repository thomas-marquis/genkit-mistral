package internal

import (
	_ "embed"
	"fmt"
	"math/rand"
	"strings"
	"unicode"
)

// GetOrZero retrieves a value of type T from a map[string]any by key.
// If the key does not exist or the value is not of type T, it returns the zero value of T.
func GetOrZero[T any](m map[string]any, key string) T {
	var zero T
	if value, ok := m[key]; ok {
		if v, ok := value.(T); ok {
			return v
		}
		return zero
	}
	return zero
}

func GetOr[T any](m map[string]any, key string, defaultValue T) T {
	if value, ok := m[key]; ok {
		if v, ok := value.(T); ok {
			return v
		}
	}
	return defaultValue
}

// GetSliceOrNil retrieves a slice of type T from a map[string]any by key.
// If the key does not exist or the value is not a slice of type T, it returns nil.
func GetSliceOrNil[T any](m map[string]any, key string) []T {
	if value, ok := m[key]; ok {
		if slice, ok := value.([]T); ok {
			return slice
		}
	}
	return nil
}

//go:embed loremispum.txt
var loremIpsumText string

var loremIpsumWords []string

func init() {
	// remove punctuation to have clean words.
	replacer := strings.NewReplacer(".", "", ",", "")
	cleanText := replacer.Replace(loremIpsumText)
	loremIpsumWords = strings.Fields(cleanText)
}

// FakeText generates a fake text with a given number of words from lorem ipsum text.
// The returned text will start with a capital letter and end with a period.
// It can generate text longer than the source lorem ipsum text.
func FakeText(wordCount int) (string, error) {
	if wordCount < 0 {
		return "", fmt.Errorf("word count cannot be negative")
	}

	if wordCount == 0 {
		return ".", nil
	}

	numWords := len(loremIpsumWords)
	if numWords == 0 {
		// This can happen if the embedded file is empty.
		return "", fmt.Errorf("lorem ipsum text is empty or not loaded")
	}

	var resultWords []string
	startIndex := rand.Intn(numWords)

	for i := 0; i < wordCount; i++ {
		resultWords = append(resultWords, loremIpsumWords[(startIndex+i)%numWords])
	}

	text := strings.Join(resultWords, " ")

	runes := []rune(text)
	runes[0] = unicode.ToUpper(runes[0])

	return string(runes) + ".", nil
}
