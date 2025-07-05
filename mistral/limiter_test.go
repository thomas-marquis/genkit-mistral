package mistral_test

import (
	"github.com/stretchr/testify/assert"
	"github.com/thomas-marquis/genkit-mistral/mistral"
	"sync"
	"testing"
	"time"
)

const (
	timeout = 10 * time.Second
	delta   = 100 * time.Millisecond
)

func timeUntilUnlocked(t *testing.T, blockingFunc func(), startedAt time.Time, elapsed *time.Duration) {
	t.Helper()
	done := make(chan bool)
	go func() {
		blockingFunc()
		done <- true
	}()

	select {
	case <-done:
		*elapsed = time.Since(startedAt)
	case <-time.After(timeout):
		t.Errorf("Function did not complete within the timeout of %v", timeout)
	}
}

func timeUntilAllUnlocked(t *testing.T, blockingFunc func(), nb int, startedAt time.Time, elapsed *time.Duration) {
	timeUntilUnlocked(t, func() {
		var wg sync.WaitGroup
		for i := 0; i < nb; i++ {
			wg.Add(1)
			go func() {
				blockingFunc()
				wg.Done()
			}()
		}
		wg.Wait()
	}, startedAt, elapsed)
}

func TestBucketCallsRateLimiter_ShouldAllowFirstCall(t *testing.T) {
	// Given
	rateLimiter := mistral.NewBucketCallsRateLimiter(10, 10, time.Second)
	defer rateLimiter.Stop()
	var elapsed time.Duration
	start := time.Now()

	// When
	timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)

	// Then
	assert.LessOrEqual(t, elapsed, 100*time.Millisecond, "First request should be allowed quickly")
}

func TestBucketCallsRateLimiter_ShouldNotAllowWhenRateIsReached(t *testing.T) {
	// Given
	rateLimiter := mistral.NewBucketCallsRateLimiter(1, 1, time.Second)
	defer rateLimiter.Stop()
	var elapsed time.Duration
	start := time.Now()

	// When & Then
	timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)
	assert.LessOrEqual(t, elapsed, 100*time.Millisecond, "First request should be allowed quickly. But took %v", elapsed)

	timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)
	assert.GreaterOrEqual(t, elapsed, 1000*time.Millisecond, "Second request should be blocked until rate is refilled, took %v", elapsed)

	timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)
	assert.GreaterOrEqual(t, elapsed, 2000*time.Millisecond, "Third request should be allowed after a second refill, took %v", elapsed)
}

func TestBucketCallsRateLimiter_ShouldBeAbleToHandleManyRequests(t *testing.T) {
	// Given
	rateLimiter := mistral.NewBucketCallsRateLimiter(5, 10, time.Second)
	defer rateLimiter.Stop()
	var elapsed time.Duration
	start := time.Now()

	// When & Then
	timeUntilAllUnlocked(t, rateLimiter.Wait, 10, start, &elapsed)
	assert.LessOrEqual(t, elapsed, 100*time.Millisecond, "1) All requests should be allowed quickly, elapsed= %v", elapsed)

	// No tokens left, next request should be denied
	timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)
	assert.GreaterOrEqual(t, elapsed, 1000*time.Millisecond, "2) Requests after exhaustion should be blocked, elapsed= %v", elapsed)
	assert.Lessf(t, elapsed, 1100*time.Millisecond, "2) Requests after exhaustion should be blocked for less than 1 seconds, elapsed= %v", elapsed)

	// After refill, some requests should be allowed again
	timeUntilAllUnlocked(t, rateLimiter.Wait, 4, start, &elapsed)
	assert.GreaterOrEqual(t, elapsed, 1000*time.Millisecond, "3) Requests after refill should be allowed quickly, elapsed= %v", elapsed)
	assert.Lessf(t, elapsed, 1100*time.Millisecond, "3) Requests after refill should be allowed for less than 1 seconds, elapsed= %v", elapsed)

	// No tokens left again
	timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)
	assert.GreaterOrEqual(t, elapsed, 2000*time.Millisecond, "4) Request after refill should be blocked again, elapsed=%v", elapsed)
	assert.Lessf(t, elapsed, 2100*time.Millisecond, "4) Request after refill should be blocked for less than 1 seconds, elapsed= %v", elapsed)
}

func TestNoneRateLimiter_ShouldAllowsAll(t *testing.T) {
	// Given
	var elapsed time.Duration
	rateLimiter := mistral.NewNoneRateLimiter()

	start := time.Now()

	// When & Then
	for i := 0; i < 100; i++ {
		timeUntilUnlocked(t, rateLimiter.Wait, start, &elapsed)
		assert.LessOrEqual(t, elapsed, 100*time.Millisecond, "Request %d should be allowed quickly, took %v", i, elapsed)
	}
}
