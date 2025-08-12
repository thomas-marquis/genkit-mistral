package internal_test

import (
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/internal"
	"strings"
	"testing"
	"unicode"
)

func Test_FakeText(t *testing.T) {
	tests := []struct {
		name      string
		wordCount int
		wantErr   bool
	}{
		{name: "10 words", wordCount: 10, wantErr: false},
		{name: "0 words", wordCount: 0, wantErr: false},
		{name: "1 word", wordCount: 1, wantErr: false},
		{name: "1000 words (more than source)", wordCount: 1000, wantErr: false},
		{name: "negative word count", wordCount: -1, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := internal.FakeText(tt.wordCount)
			if tt.wantErr {
				assert.Error(t, err, "Error expected for test case %s", tt.name)
				return
			}

			if tt.wordCount == 0 {
				assert.Equal(t, ".", got,
					"FakeText() result should be a point when word count is 0.")
				return
			}

			// Check ends with a period
			assert.True(t, strings.HasSuffix(got, "."),
				"FakeText() result should end with a period, but it doesn't. got: %q", got)

			// Check starts with uppercase
			runes := []rune(got)
			assert.True(t, unicode.IsUpper(runes[0]),
				"FakeText() result should start with an uppercase letter, but it doesn't. got: %q", got)

			// Check word count
			trimmed := strings.TrimSuffix(got, ".")
			words := strings.Fields(trimmed)
			assert.Equal(t, tt.wordCount, len(words),
				"FakeText() result should have the same number of words as the word count.")
		})
	}
}

func Test_GetOrZero_ReturnsValueWhenKeyExists(t *testing.T) {
	// Given
	testMap := map[string]interface{}{
		"one": 1,
		"two": 2,
	}
	key := "one"

	// When
	result := internal.GetOrZero[int](testMap, key)

	// Then
	assert.Equal(t, 1, result, "Expected the value associated with the existing key.")
}

func Test_GetOrZero_ReturnsZeroWhenKeyDoesNotExist(t *testing.T) {
	// Given
	testMap := map[string]any{
		"one": 1,
		"two": 2,
	}
	key := "three"

	// When
	result := internal.GetOrZero[float64](testMap, key)

	// Then
	assert.Equal(t, 0.0, result, "Expected zero value for non-existing key.")
}

func Test_GetOrZero_ReturnsZeroWhenMapIsEmpty(t *testing.T) {
	// Given
	testMap := map[string]any{}
	key := "one"

	// When
	result := internal.GetOrZero[string](testMap, key)

	// Then
	assert.Equal(t, "", result, "Expected zero value for empty map.")
}

func Test_GetOrZero_ReturnsValueWhenKeyExistsInAnyMap(t *testing.T) {
	// Given
	testMap := map[string]any{
		"one":   1,
		"two":   "two",
		"three": true,
	}
	key := "two"

	// When
	result := internal.GetOrZero[string](testMap, key)

	// Then
	assert.Equal(t, "two", result, "Expected the value associated with the existing key.")
}

func Test_GetOrZero_ReturnsZeroWhenTypeIsInvalid(t *testing.T) {
	// Given
	testMap := map[string]any{
		"one": 1,
		"two": "two",
	}
	key := "two"

	// When
	result := internal.GetOrZero[int](testMap, key)

	// Then
	assert.Equal(t, 0, result, "Expected nil (zero value for interface{}) for non-existing key.")
}

func Test_GetSliceOrNil_ReturnsZeroWhenSliceNotFound(t *testing.T) {
	// Given
	testMap := map[string]any{
		"one": 1,
		"two": 2,
	}
	key := "three"

	// When
	result := internal.GetSliceOrNil[string](testMap, key)

	// Then
	assert.Nil(t, result, "Expected zero value for non-existing key.")
}

func Test_GetSliceOrNil_ReturnsSliceWhenFound(t *testing.T) {
	// Given
	testMap := map[string]any{
		"one":   1,
		"two":   2,
		"three": []string{"a", "b", "c"},
	}
	key := "three"

	// When
	result := internal.GetSliceOrNil[string](testMap, key)

	// Then
	assert.Equal(t, []string{"a", "b", "c"}, result, "Expected zero value for non-existing key.")
}
