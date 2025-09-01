package mistralclient

import (
	"time"
)

type RateLimiter interface {
	Wait()
	Stop()
}

// BucketCallsRateLimiter implements the RateLimiter interface using a channel-based token bucket algorithm.
type BucketCallsRateLimiter struct {
	rate       int           // number of callTokens to add per time unit
	capacity   int           // maximum number of callTokens in the bucket
	callTokens chan struct{} // channel to hold callTokens
	stop       chan struct{} // channel to signal goroutine to stop
}

var _ RateLimiter = (*BucketCallsRateLimiter)(nil)

// NewBucketCallsRateLimiter creates a new token bucket rate limiter using channels and goroutines.
func NewBucketCallsRateLimiter(rate, capacity int, timeUnit time.Duration) *BucketCallsRateLimiter {
	limiter := &BucketCallsRateLimiter{
		rate:       rate,
		capacity:   capacity,
		callTokens: make(chan struct{}, capacity),
		stop:       make(chan struct{}),
	}

	// fill the bucket initially
	for i := 0; i < capacity; i++ {
		limiter.callTokens <- struct{}{}
	}

	// start a goroutine to refill the bucket periodically
	go limiter.refill(timeUnit)

	return limiter
}

// refill starts a loop that periodically adds callTokens to the bucket.
func (rl *BucketCallsRateLimiter) refill(timeUnit time.Duration) {
	ticker := time.NewTicker(timeUnit)
	defer ticker.Stop()

	for {
		select {
		case <-rl.stop:
			return
		case <-ticker.C:
			// Calculate the number of callTokens to add to reach full capacity or add by the rate
			currentTokens := len(rl.callTokens)
			tokensToAdd := rl.rate

			if currentTokens+tokensToAdd > rl.capacity {
				tokensToAdd = rl.capacity - currentTokens
			}

			for i := 0; i < tokensToAdd; i++ {
				select {
				case rl.callTokens <- struct{}{}:
				default:
					// channel full, do not add more callTokens
				}
			}
		}
	}
}

// Wait blocks until a callToken is available, effectively limiting the rate of calls.
func (rl *BucketCallsRateLimiter) Wait() {
	<-rl.callTokens
}

// Stop signals the refill goroutine to stop.
func (rl *BucketCallsRateLimiter) Stop() {
	close(rl.stop)
}

// NoneRateLimiter is a no-op implementation of RateLimiter that allows all requests without any limit.
type NoneRateLimiter struct{}

func NewNoneRateLimiter() *NoneRateLimiter {
	return &NoneRateLimiter{}
}

var _ RateLimiter = (*NoneRateLimiter)(nil)

func (n NoneRateLimiter) Wait() {}

func (n NoneRateLimiter) Stop() {}
