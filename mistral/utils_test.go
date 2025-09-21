package mistral_test

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral"
)

func Test_SanitizeToolName_ShouldRemoveInvalidChars_WhenInputContainsSpecials(t *testing.T) {
	// Given
	input := "Hello, World! 2025. #Go(lang)?"

	// When
	got := mistral.SanitizeToolName(input)

	// Then
	assert.Equal(t, "HelloWorld2025Golang", got)
}

func Test_SanitizeToolName_ShouldKeepUnderscoresAndDashes_WhenPresent(t *testing.T) {
	// Given
	input := "my_func-name__--OK"

	// When
	got := mistral.SanitizeToolName(input)

	// Then
	assert.Equal(t, "my_func-name__--OK", got)
}

func Test_SanitizeToolName_ShouldTruncate_WhenLongerThan256(t *testing.T) {
	// Given
	input := strings.Repeat("a", 300)

	// When
	got := mistral.SanitizeToolName(input)

	// Then
	assert.Len(t, got, 256)
	assert.Equal(t, strings.Repeat("a", 256), got)
}

func Test_SanitizeToolName_ShouldReturnEmpty_WhenInputEmpty(t *testing.T) {
	// Given
	input := ""

	// When
	got := mistral.SanitizeToolName(input)

	// Then
	assert.Equal(t, "", got)
}

func Test_SanitizeToolName_ShouldKeepAlnum_WhenInputIsAlnum(t *testing.T) {
	// Given
	input := "Abc123XYZ"

	// When
	got := mistral.SanitizeToolName(input)

	// Then
	assert.Equal(t, "Abc123XYZ", got)
}

func Test_SanitizeToolName_ShouldRemoveUnicode_WhenNonAsciiLetters(t *testing.T) {
	// Given
	input := "Café_été-東京-ß" // accents and non-ASCII should be removed; underscore and dash kept

	// When
	got := mistral.SanitizeToolName(input)

	// Then
	assert.Equal(t, "Caf_t_-", got)
}
